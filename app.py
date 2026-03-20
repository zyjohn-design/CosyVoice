"""
CosyVoice2 TTS 生产级 API 服务
================================

模型: CosyVoice2-0.5B (纯 zero_shot 推理, 不支持 SFT/Instruct)
框架: FastAPI + uvicorn (异步, 高并发)

说话人管理:
  speakers/ 目录下放置:
    - 参考音频: {说话人名}.wav  (3~10秒, 采样率>=16kHz)
    - 对应文本: {说话人名}.txt  (音频中说的话, 用于文本-音频对齐)
  服务启动时自动扫描并注册到模型, 推理时通过 zero_shot_spk_id 复用。

接口列表:
  GET/POST  /tts                - 统一 TTS 入口 (兼容原 app.py 参数)
  POST      /tts/zero_shot      - 上传参考音频实时克隆声音
  POST      /tts/stream         - 流式低延迟合成
  GET       /speakers           - 列出所有已注册说话人
  POST      /speakers/upload    - 上传音频注册新说话人
  DELETE    /speakers/{name}    - 删除说话人
  POST      /clear_wavs         - 清理生成的临时音频文件
  GET       /health             - 服务健康检查

性能优化 (GPU 部署):
  通过环境变量启用:
    LOAD_JIT=true   → TorchScript 加速 (+10~15%)
    LOAD_TRT=true   → TensorRT 加速 Flow Matching (+2~3x)
    LOAD_VLLM=true  → vLLM 加速 LLM decoding (+4x)
    FP16=true       → 半精度推理 (+50% 速度, -50% 显存)
"""

# ══════════════════════════════════════════════════════════════
#  依赖导入
# ══════════════════════════════════════════════════════════════

import os
import sys
import io
import re
import time
import uuid
import shutil
import logging
import asyncio
import datetime
from typing import Optional
from contextlib import asynccontextmanager
from random import random

import numpy as np
import torch
import torch._dynamo
import torchaudio
import soundfile as sf

# ── PyTorch 全局优化配置 ──
torch._dynamo.config.suppress_errors = True       # 忽略 dynamo 编译错误, 回退到 eager 模式
torch._dynamo.config.cache_size_limit = 64         # dynamo 缓存大小
torch.set_float32_matmul_precision("high")          # 允许 TF32 加速矩阵运算
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"        # 解决 macOS 上 OpenMP 重复加载问题

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ══════════════════════════════════════════════════════════════
#  配置项 (均可通过环境变量覆盖)
# ══════════════════════════════════════════════════════════════

HOST = os.getenv("HOST", "0.0.0.0")                # 监听地址
PORT = int(os.getenv("PORT", "8080"))               # 监听端口
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))  # 最大并发推理数 (防止 GPU OOM)

ROOT_DIR = os.getenv("ROOT_DIR", os.path.dirname(os.path.abspath(__file__)))
SPEAKER_DIR = os.getenv("SPEAKER_DIR", os.path.join(ROOT_DIR, "speakers"))      # 说话人文件目录
WAVS_DIR = os.getenv("WAVS_DIR", os.path.join(ROOT_DIR, "static", "wavs"))      # 生成音频输出目录
LOGS_DIR = os.getenv("LOGS_DIR", os.path.join(ROOT_DIR, "logs"))                 # 日志目录
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(ROOT_DIR, "uploads"))           # 临时上传目录
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(ROOT_DIR, "pretrained_models", "CosyVoice2-0.5B"))  # 模型路径

SAMPLE_RATE = 24000  # CosyVoice2 默认采样率, 启动后从模型获取实际值

# ── 日志配置 ──
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("cosyvoice-api")

# ══════════════════════════════════════════════════════════════
#  全局状态
# ══════════════════════════════════════════════════════════════

cosyvoice_model = None                      # CosyVoice2 模型实例
_infer_semaphore: asyncio.Semaphore = None  # 推理并发信号量
_registered_spk_ids: set = set()            # 已成功注册到模型的说话人 ID 集合

# 支持的参考音频格式
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}


# ══════════════════════════════════════════════════════════════
#  音频加载工具
# ══════════════════════════════════════════════════════════════

def _load_wav_16k(audio_path: str) -> torch.Tensor:
    """
    加载音频文件并转换为 16kHz 单声道 tensor。

    Args:
        audio_path: 音频文件路径 (支持 wav/mp3/flac 等 torchaudio 支持的格式)

    Returns:
        torch.Tensor: shape=(1, num_samples), 16kHz 采样率

    Steps:
        1. torchaudio.load 加载原始音频
        2. 多声道 → 取均值转为单声道
        3. 非 16kHz → resample 到 16kHz
    """
    # Step 1: 加载音频文件
    waveform, sr = torchaudio.load(audio_path)

    # Step 2: 多声道转单声道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Step 3: 重采样到 16kHz (CosyVoice 要求的输入采样率)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    return waveform


# ══════════════════════════════════════════════════════════════
#  说话人管理
#
#  目录结构示例:
#    speakers/
#    ├── 1111-男声.wav     ← 参考音频 (3~10秒清晰语音)
#    ├── 1111-男声.txt     ← 音频内容的文字 (必须准确对应)
#    ├── 10003-女声.wav
#    └── 10003-女声.txt
#
#  注册流程:
#    .wav + .txt → add_zero_shot_spk(prompt_text, audio, spk_id)
#    → 模型内部提取 speaker embedding 并缓存
#    → 后续推理时通过 zero_shot_spk_id=spk_id 复用
# ══════════════════════════════════════════════════════════════

def _register_audio_speaker(spk_id: str, audio_path: str) -> bool:
    """
    注册单个音频说话人到 CosyVoice 模型。

    Args:
        spk_id:     说话人唯一标识 (文件名去掉扩展名, 如 "1111-男声")
        audio_path: 参考音频文件的完整路径

    Returns:
        bool: 是否注册成功

    Steps:
        1. 查找同名 .txt 文件读取 prompt_text (音频中说的话)
        2. 尝试方式1: 传文件路径给模型 (CosyVoice3 风格)
        3. 方式1失败则尝试方式2: 加载为 16k tensor 再传 (CosyVoice2 风格)
        4. 注册成功后将 spk_id 加入 _registered_spk_ids 集合
    """
    # ── Step 1: 读取 prompt_text ──
    # prompt_text 必须是参考音频中说的话的准确文字
    # 如果不匹配, 模型内部的文本-音频对齐会出错, 导致合成内容乱码
    prompt_text = ""
    txt_path = os.path.splitext(audio_path)[0] + ".txt"
    if os.path.isfile(txt_path):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
        except Exception:
            pass

    if not prompt_text:
        logger.warning(
            f"Speaker '{spk_id}': 未找到 .txt 文件, "
            f"请创建 {txt_path} 并写入音频中的文字内容"
        )

    # ── Step 2: 方式1 - 传文件路径 (CosyVoice3 接受字符串路径) ──
    try:
        cosyvoice_model.add_zero_shot_spk(prompt_text, audio_path, spk_id)
        _registered_spk_ids.add(spk_id)
        logger.info(f"已注册说话人 '{spk_id}' (文件路径模式)")
        return True
    except Exception:
        pass  # CosyVoice2 不接受字符串路径, 继续尝试方式2

    # ── Step 3: 方式2 - 传 16k tensor (CosyVoice2 接受 tensor) ──
    try:
        prompt_speech_16k = _load_wav_16k(audio_path)
        cosyvoice_model.add_zero_shot_spk(prompt_text, prompt_speech_16k, spk_id)
        _registered_spk_ids.add(spk_id)
        logger.info(f"已注册说话人 '{spk_id}' (tensor 模式)")
        return True
    except Exception as e:
        logger.warning(f"注册说话人 '{spk_id}' 失败: {e}")
        return False


def _register_all_speakers():
    """
    启动时扫描 speakers/ 目录, 注册所有音频说话人。

    Steps:
        1. 遍历 speakers/ 下所有文件
        2. 筛选音频文件 (.wav/.mp3/.flac/.ogg)
        3. 跳过 .txt 文件 (它们是 prompt_text, 由 _register_audio_speaker 内部读取)
        4. 逐个调用 _register_audio_speaker 注册
        5. 输出注册结果统计
    """
    if not os.path.isdir(SPEAKER_DIR):
        return

    ok, total = 0, 0
    for f in sorted(os.listdir(SPEAKER_DIR)):
        stem, ext = os.path.splitext(f)
        # 只处理音频文件, .txt 文件会在注册时自动关联
        if ext.lower() in _AUDIO_EXTS:
            total += 1
            if _register_audio_speaker(stem, os.path.join(SPEAKER_DIR, f)):
                ok += 1

    logger.info(f"说话人注册完成: {ok}/{total} (目录: {SPEAKER_DIR})")
    if _registered_spk_ids:
        logger.info(f"已注册: {sorted(_registered_spk_ids)}")


# ══════════════════════════════════════════════════════════════
#  请求参数解析工具
# ══════════════════════════════════════════════════════════════

def get_param(params: dict, key: str, default, cast_type=str):
    """
    从参数字典中提取值并做类型转换。

    Args:
        params:    参数字典
        key:       参数名
        default:   默认值 (参数不存在或为空时返回)
        cast_type: 目标类型 (int/float/str)

    Returns:
        转换后的值, 或 default
    """
    val = params.get(key)
    if val is None or val == "":
        return default
    try:
        return cast_type(val)
    except (ValueError, TypeError):
        return default


async def parse_request_params(request: Request) -> dict:
    """
    统一解析 HTTP 请求参数, 兼容多种传参方式。

    支持:
        - GET query string:                    /tts?text=你好&voice=张三
        - POST application/json:               {"text": "你好", "voice": "张三"}
        - POST application/x-www-form-urlencoded: text=你好&voice=张三
        - POST multipart/form-data:            (表单上传)

    Args:
        request: FastAPI Request 对象

    Returns:
        dict: 合并后的参数字典 (query params + body params)
    """
    # Step 1: 先取 query string 参数
    params = dict(request.query_params)

    # Step 2: POST 请求再取 body 参数
    if request.method == "POST":
        ct = request.headers.get("content-type", "")
        if "application/json" in ct:
            # JSON body
            try:
                body = await request.json()
                if isinstance(body, dict):
                    params.update(body)
            except Exception:
                pass
        else:
            # form-data / x-www-form-urlencoded
            form = await request.form()
            for k, v in form.items():
                params[k] = v

    return params


# ══════════════════════════════════════════════════════════════
#  音频处理工具
# ══════════════════════════════════════════════════════════════

def audio_tensor_to_bytes(t: torch.Tensor, sr: int) -> bytes:
    """
    将音频 tensor 编码为 WAV 格式的 bytes。

    Args:
        t:  音频 tensor, shape=(channels, samples) 或 (samples,)
        sr: 采样率

    Returns:
        bytes: WAV 格式的音频数据
    """
    # 确保是 2D tensor (channels, samples)
    if t.ndim == 1:
        t = t.unsqueeze(0)

    buf = io.BytesIO()
    torchaudio.save(buf, t, sr, format="wav")
    buf.seek(0)
    return buf.read()


def collect_chunks(results: list) -> torch.Tensor:
    """
    将 CosyVoice 推理产生的多个 chunk 拼接为完整音频。

    CosyVoice 的 inference_zero_shot 返回一个生成器,
    每次 yield {"tts_speech": tensor}, 本函数将所有 chunk 拼接起来。

    Args:
        results: list of dict, 每个 dict 包含 "tts_speech" 字段

    Returns:
        torch.Tensor: shape=(1, total_samples), 完整音频

    Raises:
        HTTPException: 如果没有任何 chunk (推理失败)
    """
    chunks = []
    for item in results:
        sp = item["tts_speech"]
        # 统一为 2D: (1, samples)
        if sp.ndim == 1:
            sp = sp.unsqueeze(0)
        chunks.append(sp)

    if not chunks:
        raise HTTPException(status_code=500, detail="推理未产生音频")

    # 沿时间轴 (dim=1) 拼接所有 chunk
    return torch.cat(chunks, dim=1)


def save_to_wavs_dir(audio: torch.Tensor, sr: int, tag: str = "") -> tuple:
    """
    将音频保存到 WAVS_DIR 目录, 生成唯一文件名。

    Args:
        audio: 音频 tensor
        sr:    采样率
        tag:   文件名中的描述标签

    Returns:
        tuple: (abs_path, filename)
            - abs_path:  文件绝对路径, 如 /path/to/static/wavs/114530_xxxxx.wav
            - filename:  文件名, 如 114530_xxxxx.wav
    """
    # 生成唯一文件名: 时间戳 + 标签 + 随机数
    outname = datetime.datetime.now().strftime("%H%M%S_") + f"{tag}-{str(random())[2:7]}.wav"
    out_path = os.path.join(WAVS_DIR, outname)

    # 确保是 2D tensor
    torchaudio.save(out_path, audio if audio.ndim == 2 else audio.unsqueeze(0), sr)
    return out_path, outname


def resolve_voice(voice_raw: str, custom_voice: int = 0) -> Optional[str]:
    """
    解析 voice 参数, 返回已注册的说话人 ID。

    兼容原 app.py 的 voice 参数格式:
        voice="1111-男声"       → 直接匹配已注册说话人
        voice="1111-男声.wav"   → 去掉扩展名后匹配
        voice="1111-男声.pt"    → 去掉扩展名后匹配
        custom_voice=9999       → 覆盖 voice, 转为 "9999" 后匹配

    回退策略:
        voice 不在已注册列表中 → 使用第一个已注册的说话人

    Args:
        voice_raw:    原始 voice 参数值
        custom_voice: 自定义音色值, >0 时覆盖 voice_raw

    Returns:
        str: 已注册的 spk_id, 或 None (无任何已注册说话人)
    """
    # Step 1: custom_voice 优先级高于 voice
    if custom_voice > 0:
        voice_raw = str(custom_voice)

    # Step 2: 去掉可能的扩展名
    spk_id = voice_raw.replace(".pt", "").replace(".wav", "").replace(".csv", "")

    # Step 3: 精确匹配
    if spk_id in _registered_spk_ids:
        return spk_id

    # Step 4: 回退到第一个已注册说话人 (按名称排序)
    if _registered_spk_ids:
        fallback = sorted(_registered_spk_ids)[0]
        logger.info(f"说话人 '{voice_raw}' 未注册, 回退到 '{fallback}'")
        return fallback

    # 没有任何已注册说话人
    return None


# ══════════════════════════════════════════════════════════════
#  并发控制
# ══════════════════════════════════════════════════════════════

async def run_in_pool(fn):
    """
    在线程池中运行同步推理函数, 受信号量控制并发。

    CosyVoice 的推理是同步阻塞的 (GPU 计算),
    用 run_in_executor 放到线程池中, 避免阻塞 FastAPI 的 event loop。
    信号量限制同时推理的数量, 防止 GPU OOM。

    Args:
        fn: 无参数的 callable, 返回推理结果
    """
    async with _infer_semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn)


# ══════════════════════════════════════════════════════════════
#  应用生命周期 (启动/关闭)
# ══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 应用生命周期管理。

    启动阶段:
        1. 初始化并发信号量
        2. 创建必要的目录
        3. 检查 ffmpeg 是否安装
        4. 加载 CosyVoice2 模型 (支持 GPU 优化参数)
        5. 扫描 speakers/ 目录, 注册所有说话人
        6. 用第一个已注册说话人做预热推理 (初始化 CUDA kernel)

    关闭阶段:
        清理 GPU 显存
    """
    global cosyvoice_model, _infer_semaphore, SAMPLE_RATE

    # ── Step 1: 初始化并发信号量 ──
    _infer_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    # ── Step 2: 创建目录 ──
    for d in [SPEAKER_DIR, WAVS_DIR, LOGS_DIR, UPLOAD_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Step 3: 检查 ffmpeg ──
    if not shutil.which("ffmpeg"):
        logger.error("ffmpeg 未安装! 请先安装: brew install ffmpeg (macOS) 或 apt install ffmpeg (Linux)")
        raise RuntimeError("ffmpeg is required")

    # ── Step 4: 加载 CosyVoice2 模型 ──
    # 添加 third_party 到 Python path
    sys.path.insert(0, os.path.join(ROOT_DIR, "third_party", "Matcha-TTS"))
    sys.path.insert(0, ROOT_DIR)

    # ── patch wetext 的 snapshot_download 为离线模式 ──
    # wetext 内部调用 snapshot_download("pengzhendong/wetext") 会联网检查版本
    # 如果本地已有缓存, 直接返回缓存路径, 不联网
    _wetext_cache = os.path.expanduser("~/.cache/modelscope/hub/pengzhendong/wetext")
    if os.path.isdir(_wetext_cache):
        try:
            import modelscope.hub.snapshot_download as _sd_module
            _original_download = _sd_module.snapshot_download
            def _offline_snapshot_download(model_id, *args, **kwargs):
                if "wetext" in model_id:
                    logger.info(f"wetext 使用离线缓存: {_wetext_cache}")
                    return _wetext_cache
                return _original_download(model_id, *args, **kwargs)
            _sd_module.snapshot_download = _offline_snapshot_download
            # 同时 patch modelscope 顶层导入
            import modelscope
            if hasattr(modelscope, 'snapshot_download'):
                modelscope.snapshot_download = _offline_snapshot_download
            logger.info("已启用 wetext 离线模式")
        except Exception as e:
            logger.warning(f"patch wetext 离线模式失败: {e}")

    logger.info(f"正在加载 CosyVoice 模型: {MODEL_DIR} ...")
    t0 = time.time()

    # 从环境变量读取 GPU 优化参数
    load_jit = os.getenv("LOAD_JIT", "false").lower() in ("true", "1")    # TorchScript 优化
    load_trt = os.getenv("LOAD_TRT", "false").lower() in ("true", "1")    # TensorRT 加速
    load_vllm = os.getenv("LOAD_VLLM", "false").lower() in ("true", "1")  # vLLM 加速
    fp16 = os.getenv("FP16", "false").lower() in ("true", "1")            # 半精度推理

    from cosyvoice.cli.cosyvoice import AutoModel
    cosyvoice_model = AutoModel(
        model_dir=MODEL_DIR,
        load_jit=load_jit,
        load_trt=load_trt,
        load_vllm=load_vllm,
        fp16=fp16,
    )
    SAMPLE_RATE = cosyvoice_model.sample_rate
    logger.info(
        f"模型加载完成 ({time.time()-t0:.1f}s), "
        f"采样率={SAMPLE_RATE}, jit={load_jit}, trt={load_trt}, vllm={load_vllm}, fp16={fp16}"
    )

    # ── Step 5: 注册说话人 ──
    _register_all_speakers()

    # ── Step 6: 预热推理 ──
    # 第一次推理会触发 CUDA kernel 编译/JIT 编译等, 耗时较长
    # 预热后后续请求的首次延迟会大幅降低
    if _registered_spk_ids:
        try:
            spk = sorted(_registered_spk_ids)[0]
            logger.info(f"预热推理 (说话人='{spk}') ...")
            for _ in cosyvoice_model.inference_zero_shot(
                "你好", "", "", zero_shot_spk_id=spk, stream=False
            ):
                pass
            logger.info("预热完成")
        except Exception as e:
            logger.warning(f"预热失败 (不影响服务): {e}")

    # ── 应用运行期 ──
    yield

    # ── 关闭阶段: 清理 GPU 显存 ──
    logger.info("服务关闭中 ...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── 创建 FastAPI 应用 ──
app = FastAPI(
    title="CosyVoice2 TTS API",
    version="2.0.0",
    description="CosyVoice2-0.5B 语音合成服务, 支持 zero_shot 声音克隆",
    lifespan=lifespan,
)

# ── 挂载静态文件目录, 生成的音频可通过 http://host:port/static/wavs/xxx.wav 访问 ──
os.makedirs("static/wavs", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ══════════════════════════════════════════════════════════════
#  GET/POST /tts — 统一 TTS 入口
# ══════════════════════════════════════════════════════════════

@app.api_route("/tts", methods=["GET", "POST"])
async def tts(request: Request):
    """
    TTS 语音合成统一入口。

    兼容原 app.py (ChatTTS) 的所有参数, 所有推理走 CosyVoice2 zero_shot。

    === 请求参数 ===
    text (str, 必须):         要合成的文本
    voice (str, 默认"2222"):  说话人名称。匹配 speakers/ 下已注册的说话人,
                              不存在则回退到第一个已注册说话人
    custom_voice (int, 默认0): >0 时覆盖 voice 参数
    speed (int, 默认5):       语速 1~9, 其中 5=正常速度 (映射为 CosyVoice speed=1.0)
    wav (int, 默认0):         1=直接返回 WAV 音频文件, 0=返回 JSON
    is_stream (int, 默认0):   1=流式返回 chunked PCM (16bit, mono, SAMPLE_RATE Hz)
    text_seed (int, 默认42):  随机种子, 固定韵律和语调

    以下参数保留兼容, 当前模型未直接使用:
    prompt, temperature, top_p, top_k, skip_refine,
    refine_max_new_token, infer_max_new_token, uv_break

    === 返回 ===
    is_stream=1:
        StreamingResponse, media_type="audio/pcm"
        Headers: X-Sample-Rate, X-Channels, X-Bit-Depth

    wav=1:
        FileResponse, media_type="audio/x-wav"

    wav=0 (默认):
        JSON:
        {
            "code": 0, "msg": "ok",
            "audio_files": [{
                "filename": "/abs/path.wav",
                "url": "http://host/static/wavs/xxx.wav",
                "relative_url": "/static/wavs/xxx.wav",
                "inference_time": 1.23,
                "audio_duration": 3.45
            }],
            "filename": "...", "url": "...", "relative_url": "..."
        }
    """
    # ── Step 1: 解析请求参数 ──
    params = await parse_request_params(request)

    text = get_param(params, "text", "", str).strip()
    if not text:
        return JSONResponse({"code": 1, "msg": "text params lost"})

    custom_voice = get_param(params, "custom_voice", 0, int)
    voice_raw = get_param(params, "voice", "1111-男声", str)
    speed = get_param(params, "speed", 5, int)
    text_seed = get_param(params, "text_seed", 42, int)
    wav_flag = get_param(params, "wav", 0, int)
    is_stream = get_param(params, "is_stream", 0, int)

    # ── Step 2: 语速映射 ──
    # 原 app.py: speed=1~9, 5=正常
    # CosyVoice: speed 是倍率, 1.0=正常
    # 映射: speed/5 → 1=0.2x慢, 5=1.0x正常, 9=1.8x快
    cosyvoice_speed = speed / 5.0 if speed > 0 else 1.0

    # ── Step 3: 解析说话人 ──
    spk_id = resolve_voice(voice_raw, custom_voice)
    if not spk_id:
        return JSONResponse({"code": 1, "msg": "没有已注册的说话人, 请先在 speakers/ 目录放置参考音频"})

    logger.info(f"[/tts] voice={voice_raw} → spk_id={spk_id}, 文本长度={len(text)}, stream={is_stream}")

    # ── Step 4: 设置随机种子 (保证可重复) ──
    if text_seed > 0:
        torch.manual_seed(text_seed)

    # ── Step 5a: 流式模式 (is_stream=1) ──
    if is_stream == 1:
        async def audio_stream():
            """
            流式生成器: 逐 chunk 返回 PCM 音频数据。
            CosyVoice 每生成一段语音就 yield 一个 chunk,
            客户端可以边接收边播放, 实现低延迟。
            """
            try:
                gen = cosyvoice_model.inference_zero_shot(
                    text, "", "",
                    zero_shot_spk_id=spk_id,
                    stream=True,
                    speed=cosyvoice_speed,
                )
                for item in gen:
                    sp = item["tts_speech"]
                    if sp.ndim == 1:
                        sp = sp.unsqueeze(0)
                    # float32 tensor → 16bit PCM bytes
                    yield (sp.squeeze().numpy() * 32767).astype(np.int16).tobytes()
            except Exception as e:
                logger.error(f"流式推理错误: {e}", exc_info=True)

        return StreamingResponse(
            audio_stream(),
            media_type="audio/pcm",
            headers={
                "X-Sample-Rate": str(SAMPLE_RATE),
                "X-Channels": "1",
                "X-Bit-Depth": "16",
            },
        )

    # ── Step 5b: 非流式模式 ──
    t0 = time.time()
    try:
        # 在线程池中执行推理, 避免阻塞 event loop
        results = await run_in_pool(
            lambda: list(cosyvoice_model.inference_zero_shot(
                text, "", "",
                zero_shot_spk_id=spk_id,
                stream=False,
                speed=cosyvoice_speed,
            ))
        )
    except Exception as e:
        logger.error(f"推理错误: {e}", exc_info=True)
        return JSONResponse({"code": 1, "msg": str(e)})

    inference_time = round(time.time() - t0, 2)

    # ── Step 6: 拼接音频 chunk ──
    full_audio = collect_chunks(results)

    # ── Step 7: 保存到文件 ──
    tag = f"use{inference_time}s-voice{spk_id}-textlen{len(text)}"
    out_path, outname = save_to_wavs_dir(full_audio, SAMPLE_RATE, tag)

    # ── Step 8: 计算音频时长 ──
    try:
        audio_duration = round(sf.info(out_path).duration, 2)
    except Exception:
        audio_duration = round(full_audio.shape[1] / SAMPLE_RATE, 2)

    # ── Step 9: 清理 GPU 显存 ──
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Step 10a: wav=1 → 直接返回音频文件 ──
    if wav_flag > 0:
        return FileResponse(out_path, media_type="audio/x-wav")

    # ── Step 10b: wav=0 → 返回 JSON (兼容原 app.py / pyVideoTrans) ──
    relative_url = f"/static/wavs/{outname}"
    host = request.headers.get("host", "localhost")
    result = {
        "code": 0,
        "msg": "ok",
        "audio_files": [{
            "filename": out_path,
            "url": f"http://{host}{relative_url}",
            "relative_url": relative_url,
            "inference_time": inference_time,
            "audio_duration": audio_duration,
        }],
        # 顶层字段: 兼容 pyVideoTrans 直接读取
        "filename": out_path,
        "url": f"http://{host}{relative_url}",
        "relative_url": relative_url,
    }
    return JSONResponse(result)


# ══════════════════════════════════════════════════════════════
#  POST /tts/zero_shot — 上传参考音频实时克隆
# ══════════════════════════════════════════════════════════════

@app.post("/tts/zero_shot")
async def tts_zero_shot(
    text: str = Form(..., description="要合成的文本"),
    prompt_text: str = Form(..., description="参考音频中说的话 (必须准确)"),
    prompt_audio: UploadFile = File(..., description="参考音频文件 (wav, >=16kHz, 3~10秒)"),
    speed: float = Form(default=1.0, description="语速倍率, 1.0=正常"),
    stream: bool = Form(default=False, description="是否流式返回"),
    seed: int = Form(default=0, description="随机种子, 0=随机"),
    save_speaker: Optional[str] = Form(default=None, description="保存为说话人的名称, 为空则不保存"),
):
    """
    上传参考音频实时克隆声音。

    与 /tts 的区别: 不需要预注册说话人, 直接上传参考音频即可。
    可选通过 save_speaker 参数将说话人保存到 speakers/ 目录供后续复用。

    Returns:
        StreamingResponse: WAV 格式音频
        Headers: X-Inference-Time, X-Audio-Duration, X-Saved-Speaker (如有保存)
    """
    # ── Step 1: 保存上传的音频到临时文件 ──
    audio_bytes = await prompt_audio.read()
    tmp_path = os.path.join(UPLOAD_DIR, f"prompt_{uuid.uuid4().hex[:8]}.wav")
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)

    # ── Step 2: 设置随机种子 ──
    if seed > 0:
        torch.manual_seed(seed)

    # ── Step 3: 执行 zero_shot 推理 ──
    t0 = time.time()
    try:
        prompt_speech_16k = _load_wav_16k(tmp_path)
        results = await run_in_pool(
            lambda: list(cosyvoice_model.inference_zero_shot(
                text, prompt_text, prompt_speech_16k,
                stream=stream, speed=speed,
            ))
        )
    finally:
        # 清理临时文件
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # ── Step 4: 拼接音频 ──
    audio = collect_chunks(results)
    inference_time = round(time.time() - t0, 2)

    # ── Step 5: 可选保存说话人 ──
    saved = None
    if save_speaker:
        spk_name = re.sub(r'[^\w\-]', '_', save_speaker.strip())
        if spk_name:
            # 保存参考音频到 speakers/ 目录
            wav_path = os.path.join(SPEAKER_DIR, f"{spk_name}.wav")
            t = prompt_speech_16k.unsqueeze(0) if prompt_speech_16k.ndim == 1 else prompt_speech_16k
            torchaudio.save(wav_path, t, 16000)

            # 保存 prompt_text 到同名 .txt
            txt_path = os.path.join(SPEAKER_DIR, f"{spk_name}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(prompt_text)

            # 注册到模型
            cosyvoice_model.add_zero_shot_spk(prompt_text, prompt_speech_16k, spk_name)
            _registered_spk_ids.add(spk_name)
            saved = f"{spk_name}.wav"

    # ── Step 6: 返回音频 ──
    headers = {
        "X-Inference-Time": str(inference_time),
        "X-Audio-Duration": str(round(audio.shape[1] / SAMPLE_RATE, 2)),
    }
    if saved:
        headers["X-Saved-Speaker"] = saved

    return StreamingResponse(
        io.BytesIO(audio_tensor_to_bytes(audio, SAMPLE_RATE)),
        media_type="audio/wav",
        headers=headers,
    )


# ══════════════════════════════════════════════════════════════
#  POST /tts/stream — 流式低延迟合成
# ══════════════════════════════════════════════════════════════

@app.post("/tts/stream")
async def tts_stream(
    text: str = Form(..., description="要合成的文本"),
    voice: str = Form(default="", description="已注册说话人名称"),
    prompt_text: Optional[str] = Form(default=None, description="(实时克隆) 参考音频文本"),
    prompt_audio: Optional[UploadFile] = File(default=None, description="(实时克隆) 参考音频文件"),
    speed: float = Form(default=1.0, description="语速倍率"),
    seed: int = Form(default=0, description="随机种子"),
):
    """
    流式低延迟合成, 边生成边返回 PCM 音频块。

    两种用法:
        1. 指定 voice (已注册说话人): 直接用 zero_shot_spk_id
        2. 上传 prompt_audio + prompt_text: 实时克隆

    Returns:
        StreamingResponse: raw PCM (16bit, mono, SAMPLE_RATE Hz)
        客户端播放: ffmpeg -f s16le -ar 24000 -ac 1 -i stream.pcm output.wav
    """

    async def audio_stream():
        """流式生成器"""
        if seed > 0:
            torch.manual_seed(seed)

        prompt_audio_path = None
        try:
            # ── 方式1: 上传参考音频, 实时克隆 ──
            if prompt_audio and prompt_text:
                ab = await prompt_audio.read()
                prompt_audio_path = os.path.join(UPLOAD_DIR, f"stream_{uuid.uuid4().hex[:8]}.wav")
                with open(prompt_audio_path, "wb") as f:
                    f.write(ab)
                gen = cosyvoice_model.inference_zero_shot(
                    text, prompt_text, _load_wav_16k(prompt_audio_path),
                    stream=True, speed=speed,
                )
            else:
                # ── 方式2: 已注册说话人 ──
                spk_id = resolve_voice(voice)
                if not spk_id:
                    return
                gen = cosyvoice_model.inference_zero_shot(
                    text, "", "",
                    zero_shot_spk_id=spk_id,
                    stream=True, speed=speed,
                )

            # 逐 chunk 输出 PCM 数据
            for item in gen:
                sp = item["tts_speech"]
                if sp.ndim == 1:
                    sp = sp.unsqueeze(0)
                # float32 → 16bit PCM
                yield (sp.squeeze().numpy() * 32767).astype(np.int16).tobytes()

        finally:
            # 清理临时文件
            if prompt_audio_path:
                try:
                    os.remove(prompt_audio_path)
                except OSError:
                    pass

    return StreamingResponse(
        audio_stream(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Channels": "1",
            "X-Bit-Depth": "16",
        },
    )


# ══════════════════════════════════════════════════════════════
#  GET /speakers — 列出说话人
# ══════════════════════════════════════════════════════════════

@app.get("/speakers")
async def list_speakers():
    """
    列出所有说话人。

    Returns:
        JSON:
        {
            "code": 0, "msg": "ok",
            "speakers": ["1111-男声.wav", "10003-女声.wav"],  // speakers/ 下的音频文件
            "registered": ["1111-男声", "10003-女声"]          // 已成功注册到模型的 spk_id
        }
    """
    custom = []
    if os.path.isdir(SPEAKER_DIR):
        custom = sorted([
            f for f in os.listdir(SPEAKER_DIR)
            if os.path.splitext(f)[1].lower() in _AUDIO_EXTS
        ])
    return JSONResponse({
        "code": 0, "msg": "ok",
        "speakers": custom,
        "registered": sorted(_registered_spk_ids),
    })


# ══════════════════════════════════════════════════════════════
#  POST /speakers/upload — 上传注册说话人
# ══════════════════════════════════════════════════════════════

@app.post("/speakers/upload")
async def upload_speaker(
    name: str = Form(..., description="说话人名称 (将作为 spk_id)"),
    audio: UploadFile = File(..., description="参考音频 (wav, >=16kHz, 3~10秒)"),
    prompt_text: str = Form(default="", description="音频中说的话 (必须准确, 影响克隆质量)"),
):
    """
    上传参考音频注册新说话人。

    保存文件到 speakers/ 目录:
        {name}.wav  — 参考音频
        {name}.txt  — prompt_text (如有提供)

    注册成功后可通过 /tts?voice={name} 使用。

    Returns:
        JSON: {"code": 0, "msg": "ok", "speaker": "{name}.wav"}
    """
    # Step 1: 清理名称 (去除特殊字符)
    spk_name = re.sub(r'[^\w\-]', '_', name.strip())
    if not spk_name:
        return JSONResponse({"code": 1, "msg": "Invalid speaker name"})

    # Step 2: 保存音频文件
    audio_bytes = await audio.read()
    wav_path = os.path.join(SPEAKER_DIR, f"{spk_name}.wav")
    with open(wav_path, "wb") as f:
        f.write(audio_bytes)

    # Step 3: 保存 prompt_text (如有)
    if prompt_text:
        txt_path = os.path.join(SPEAKER_DIR, f"{spk_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(prompt_text)

    # Step 4: 注册到模型
    try:
        _register_audio_speaker(spk_name, wav_path)
    except Exception as e:
        return JSONResponse({"code": 1, "msg": str(e)})

    return JSONResponse({"code": 0, "msg": "ok", "speaker": f"{spk_name}.wav"})


# ══════════════════════════════════════════════════════════════
#  DELETE /speakers/{name} — 删除说话人
# ══════════════════════════════════════════════════════════════

@app.delete("/speakers/{name}")
async def delete_speaker(name: str):
    """
    删除说话人及其关联文件 (.wav + .txt)。

    Args:
        name: 说话人名称 (URL path 参数), 如 "1111-男声"

    Returns:
        JSON: {"code": 0, "msg": "ok", "deleted": ["1111-男声.wav", "1111-男声.txt"]}

    Note:
        删除文件后从 _registered_spk_ids 中移除,
        但模型内存中的 embedding 无法热移除, 需重启服务才能完全清理。
    """
    # Step 1: 去除扩展名, 获取纯名称
    bare = os.path.basename(name)
    for ext in (*_AUDIO_EXTS, ".txt"):
        bare = bare.replace(ext, "")

    # Step 2: 删除所有关联文件
    deleted = []
    for ext in (*_AUDIO_EXTS, ".txt"):
        fp = os.path.join(SPEAKER_DIR, f"{bare}{ext}")
        if os.path.isfile(fp):
            os.remove(fp)
            deleted.append(f"{bare}{ext}")

    if not deleted:
        return JSONResponse({"code": 1, "msg": f"说话人 '{bare}' 不存在"}, status_code=404)

    # Step 3: 从注册集合中移除
    _registered_spk_ids.discard(bare)

    return JSONResponse({"code": 0, "msg": "ok", "deleted": deleted})


# ══════════════════════════════════════════════════════════════
#  POST /clear_wavs — 清理生成的临时音频
# ══════════════════════════════════════════════════════════════

@app.post("/clear_wavs")
async def clear_wavs():
    """
    清理 WAVS_DIR (static/wavs/) 下的所有文件。
    用于释放磁盘空间, 不影响已注册的说话人。

    Returns:
        JSON: {"code": 0, "msg": "Cleared 42 files"}
    """
    count = 0
    for f in os.listdir(WAVS_DIR):
        fp = os.path.join(WAVS_DIR, f)
        if os.path.isfile(fp):
            os.remove(fp)
            count += 1
    return JSONResponse({"code": 0, "msg": f"Cleared {count} files"})


# ══════════════════════════════════════════════════════════════
#  GET /health — 健康检查
# ══════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """
    服务健康检查。

    Returns:
        JSON: 包含模型状态、采样率、并发数、已注册说话人、GPU 信息
    """
    gpu = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total_mb": round(props.total_memory / 1024**2),
            "gpu_memory_used_mb": round(torch.cuda.memory_allocated(0) / 1024**2),
        }
    return JSONResponse({
        "code": 0,
        "msg": "ok",
        "status": "healthy",
        "model_dir": MODEL_DIR,
        "sample_rate": SAMPLE_RATE,
        "max_concurrency": MAX_CONCURRENCY,
        "registered_speakers": sorted(_registered_spk_ids),
        "gpu": gpu,
    })


# ══════════════════════════════════════════════════════════════
#  启动入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        workers=1,              # GPU 推理只能单 worker
        log_level="info",
        timeout_keep_alive=300,  # 长连接超时 5 分钟
    )