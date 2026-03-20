"""
CosyVoice 全模型兼容 TTS API 服务
====================================

自动检测模型类型, 适配全系列 CosyVoice 模型:

  ┌──────────────────────────┬──────────┬─────┬──────────┬──────────────┐
  │ 模型                      │ zero_shot│ sft │ instruct │ cross_lingual│
  ├──────────────────────────┼──────────┼─────┼──────────┼──────────────┤
  │ CosyVoice-300M           │ ✅       │ ❌  │ ❌       │ ✅           │
  │ CosyVoice-300M-SFT       │ ❌       │ ✅  │ ❌       │ ❌           │
  │ CosyVoice-300M-Instruct  │ ❌       │ ✅  │ ✅       │ ❌           │
  │ CosyVoice2-0.5B          │ ✅       │ ❌  │ ✅(v2)   │ ✅           │
  │ Fun-CosyVoice3-0.5B      │ ✅       │ ❌  │ ✅       │ ✅           │
  └──────────────────────────┴──────────┴─────┴──────────┴──────────────┘

启动时自动检测:
  1. 通过 AutoModel 加载模型
  2. 检测模型类实例 (CosyVoice / CosyVoice2 / CosyVoice3)
  3. 探测可用方法 (inference_sft / inference_zero_shot / inference_instruct 等)
  4. 根据能力启用/禁用对应接口

接口列表:
  GET/POST  /tts                - 统一入口 (自动选择最佳推理模式)
  POST      /tts/sft            - SFT 预设音色 (仅 SFT/Instruct 模型)
  POST      /tts/zero_shot      - 零样本声音克隆 (仅 300M/V2/V3 模型)
  POST      /tts/instruct       - 自然语言指令控制 (仅 Instruct/V2/V3 模型)
  POST      /tts/stream         - 流式低延迟合成
  GET       /speakers           - 列出说话人
  POST      /speakers/upload    - 上传音频注册说话人
  DELETE    /speakers/{name}    - 删除说话人
  GET       /health             - 健康检查 (含模型能力信息)
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

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 64
torch.set_float32_matmul_precision("high")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ══════════════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════════════

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))

ROOT_DIR = os.getenv("ROOT_DIR", os.path.dirname(os.path.abspath(__file__)))
SPEAKER_DIR = os.getenv("SPEAKER_DIR", os.path.join(ROOT_DIR, "speakers"))
WAVS_DIR = os.getenv("WAVS_DIR", os.path.join(ROOT_DIR, "static", "wavs"))
LOGS_DIR = os.getenv("LOGS_DIR", os.path.join(ROOT_DIR, "logs"))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(ROOT_DIR, "uploads"))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(ROOT_DIR, "pretrained_models", "CosyVoice2-0.5B"))

SAMPLE_RATE = 24000

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("cosyvoice-api")

# ══════════════════════════════════════════════════════════════
#  全局状态
# ══════════════════════════════════════════════════════════════

cosyvoice_model = None
_infer_semaphore: asyncio.Semaphore = None
_registered_spk_ids: set = set()
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}

# ── 模型能力检测结果 (启动时填充) ──
_model_caps = {
    "model_type": "unknown",     # "CosyVoice" / "CosyVoice2" / "CosyVoice3"
    "has_sft": False,            # 是否支持 inference_sft
    "has_zero_shot": False,      # 是否支持 inference_zero_shot
    "has_instruct": False,       # 是否支持 inference_instruct / inference_instruct2
    "has_cross_lingual": False,  # 是否支持 inference_cross_lingual
    "sft_speakers": [],          # SFT 内置音色列表 (如 ["中文女","中文男",...])
    "instruct_fn_name": None,    # instruct 方法名 ("inference_instruct2" 或 "inference_instruct")
}


def _detect_model_capabilities():
    """
    探测模型实例的类型和可用推理方法, 填充 _model_caps。

    Steps:
        1. 通过类名判断模型版本 (CosyVoice / CosyVoice2 / CosyVoice3)
        2. 检查各 inference_xxx 方法是否存在
        3. 尝试 list_available_spks() 获取 SFT 音色列表
        4. 确定 instruct 方法名 (v2 用 inference_instruct2, v1 用 inference_instruct)
    """
    global _model_caps

    # Step 1: 判断模型类型
    class_name = type(cosyvoice_model).__name__
    if "CosyVoice3" in class_name:
        _model_caps["model_type"] = "CosyVoice3"
    elif "CosyVoice2" in class_name:
        _model_caps["model_type"] = "CosyVoice2"
    else:
        _model_caps["model_type"] = "CosyVoice"

    # Step 2: 探测各推理方法
    _model_caps["has_zero_shot"] = hasattr(cosyvoice_model, "inference_zero_shot")
    _model_caps["has_sft"] = hasattr(cosyvoice_model, "inference_sft")
    _model_caps["has_cross_lingual"] = hasattr(cosyvoice_model, "inference_cross_lingual")

    # Step 3: 探测 instruct (v2 优先)
    if hasattr(cosyvoice_model, "inference_instruct2"):
        _model_caps["has_instruct"] = True
        _model_caps["instruct_fn_name"] = "inference_instruct2"
    elif hasattr(cosyvoice_model, "inference_instruct"):
        _model_caps["has_instruct"] = True
        _model_caps["instruct_fn_name"] = "inference_instruct"

    # Step 4: 获取 SFT 音色列表
    try:
        spks = cosyvoice_model.list_available_spks()
        if spks:
            _model_caps["sft_speakers"] = spks
            _model_caps["has_sft"] = True
    except Exception:
        pass

    logger.info(f"模型类型: {_model_caps['model_type']}")
    logger.info(f"模型能力: sft={_model_caps['has_sft']}, zero_shot={_model_caps['has_zero_shot']}, "
                f"instruct={_model_caps['has_instruct']}, cross_lingual={_model_caps['has_cross_lingual']}")
    if _model_caps["sft_speakers"]:
        logger.info(f"SFT 内置音色: {_model_caps['sft_speakers']}")


# ══════════════════════════════════════════════════════════════
#  音频加载
# ══════════════════════════════════════════════════════════════

def _load_wav_16k(audio_path: str) -> torch.Tensor:
    """加载音频 → 16kHz mono tensor。"""
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    return waveform


# ══════════════════════════════════════════════════════════════
#  说话人管理 (仅 zero_shot 模型需要)
# ══════════════════════════════════════════════════════════════

def _register_audio_speaker(spk_id: str, audio_path: str) -> bool:
    """注册音频说话人 (.wav + .txt → add_zero_shot_spk)。"""
    if not _model_caps["has_zero_shot"]:
        return False

    # 读取 prompt_text
    prompt_text = ""
    txt_path = os.path.splitext(audio_path)[0] + ".txt"
    if os.path.isfile(txt_path):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
        except Exception:
            pass
    if not prompt_text:
        logger.warning(f"Speaker '{spk_id}': 未找到 .txt, 请创建 {txt_path}")

    # 方式1: 文件路径
    try:
        cosyvoice_model.add_zero_shot_spk(prompt_text, audio_path, spk_id)
        _registered_spk_ids.add(spk_id)
        logger.info(f"已注册说话人 '{spk_id}' (filepath)")
        return True
    except Exception:
        pass

    # 方式2: 16k tensor
    try:
        cosyvoice_model.add_zero_shot_spk(prompt_text, _load_wav_16k(audio_path), spk_id)
        _registered_spk_ids.add(spk_id)
        logger.info(f"已注册说话人 '{spk_id}' (tensor)")
        return True
    except Exception as e:
        logger.warning(f"注册说话人 '{spk_id}' 失败: {e}")
        return False


def _register_all_speakers():
    """扫描 speakers/ 下音频文件并注册。"""
    if not os.path.isdir(SPEAKER_DIR):
        return
    if not _model_caps["has_zero_shot"]:
        logger.info("当前模型不支持 zero_shot, 跳过说话人注册")
        return

    ok, total = 0, 0
    for f in sorted(os.listdir(SPEAKER_DIR)):
        stem, ext = os.path.splitext(f)
        if ext.lower() in _AUDIO_EXTS:
            total += 1
            if _register_audio_speaker(stem, os.path.join(SPEAKER_DIR, f)):
                ok += 1
    logger.info(f"说话人注册: {ok}/{total} (目录: {SPEAKER_DIR})")
    if _registered_spk_ids:
        logger.info(f"已注册: {sorted(_registered_spk_ids)}")


# ══════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════

def get_param(params: dict, key: str, default, cast_type=str):
    val = params.get(key)
    if val is None or val == "":
        return default
    try:
        return cast_type(val)
    except (ValueError, TypeError):
        return default


async def parse_request_params(request: Request) -> dict:
    params = dict(request.query_params)
    if request.method == "POST":
        ct = request.headers.get("content-type", "")
        if "application/json" in ct:
            try:
                body = await request.json()
                if isinstance(body, dict):
                    params.update(body)
            except Exception:
                pass
        else:
            form = await request.form()
            for k, v in form.items():
                params[k] = v
    return params


def audio_tensor_to_bytes(t: torch.Tensor, sr: int) -> bytes:
    if t.ndim == 1:
        t = t.unsqueeze(0)
    buf = io.BytesIO()
    torchaudio.save(buf, t, sr, format="wav")
    buf.seek(0)
    return buf.read()


def collect_chunks(results: list) -> torch.Tensor:
    chunks = []
    for item in results:
        sp = item["tts_speech"]
        if sp.ndim == 1:
            sp = sp.unsqueeze(0)
        chunks.append(sp)
    if not chunks:
        raise HTTPException(status_code=500, detail="推理未产生音频")
    return torch.cat(chunks, dim=1)


def save_to_wavs_dir(audio: torch.Tensor, sr: int, tag: str = "") -> tuple:
    outname = datetime.datetime.now().strftime("%H%M%S_") + f"{tag}-{str(random())[2:7]}.wav"
    out_path = os.path.join(WAVS_DIR, outname)
    torchaudio.save(out_path, audio if audio.ndim == 2 else audio.unsqueeze(0), sr)
    return out_path, outname


def resolve_voice(voice_raw: str, custom_voice: int = 0) -> dict:
    """
    解析 voice 参数, 根据模型能力选择最佳推理模式。

    Returns:
        dict:
            {"mode": "zero_shot_spk", "spk_id": "..."}   → 已注册说话人
            {"mode": "sft", "voice": "中文女"}             → SFT 预设音色
            {"mode": "fallback_sft", "voice": "中文女"}    → 兜底 SFT
            {"mode": "none"}                               → 无可用模式
    """
    if custom_voice > 0:
        voice_raw = str(custom_voice)

    spk_id = voice_raw.replace(".pt", "").replace(".wav", "").replace(".csv", "")

    # 优先: 匹配已注册的 zero_shot 说话人
    if spk_id in _registered_spk_ids:
        return {"mode": "zero_shot_spk", "spk_id": spk_id}

    # 其次: 匹配 SFT 内置音色名
    if _model_caps["has_sft"] and spk_id in _model_caps["sft_speakers"]:
        return {"mode": "sft", "voice": spk_id}

    # 回退: 有已注册说话人 → 用第一个
    if _registered_spk_ids:
        fallback = sorted(_registered_spk_ids)[0]
        logger.info(f"Voice '{voice_raw}' 未找到, 回退到已注册说话人 '{fallback}'")
        return {"mode": "zero_shot_spk", "spk_id": fallback}

    # 回退: 有 SFT → 用第一个内置音色
    if _model_caps["has_sft"] and _model_caps["sft_speakers"]:
        fallback = _model_caps["sft_speakers"][0]
        logger.info(f"Voice '{voice_raw}' 未找到, 回退到 SFT 音色 '{fallback}'")
        return {"mode": "sft", "voice": fallback}

    return {"mode": "none"}


async def run_in_pool(fn):
    async with _infer_semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn)


# ══════════════════════════════════════════════════════════════
#  核心推理分发
# ══════════════════════════════════════════════════════════════

def _do_inference(text: str, resolved: dict, speed: float, stream: bool) -> list:
    """
    根据 resolved 模式调用对应的推理方法。

    Args:
        text:     要合成的文本
        resolved: resolve_voice() 返回的 dict
        speed:    CosyVoice 语速倍率
        stream:   是否流式

    Returns:
        list: 推理结果列表 [{"tts_speech": tensor}, ...]
    """
    mode = resolved["mode"]

    if mode == "zero_shot_spk":
        # 已注册说话人 → zero_shot with spk_id
        return list(cosyvoice_model.inference_zero_shot(
            text, "", "",
            zero_shot_spk_id=resolved["spk_id"],
            stream=stream, speed=speed,
        ))

    elif mode == "sft":
        # SFT 预设音色
        return list(cosyvoice_model.inference_sft(
            text, resolved["voice"],
            stream=stream, speed=speed,
        ))

    else:
        raise ValueError(f"没有可用的说话人或 SFT 音色")


def _do_inference_generator(text: str, resolved: dict, speed: float):
    """
    流式版本: 返回生成器而非列表。
    """
    mode = resolved["mode"]

    if mode == "zero_shot_spk":
        return cosyvoice_model.inference_zero_shot(
            text, "", "",
            zero_shot_spk_id=resolved["spk_id"],
            stream=True, speed=speed,
        )
    elif mode == "sft":
        return cosyvoice_model.inference_sft(
            text, resolved["voice"],
            stream=True, speed=speed,
        )
    else:
        raise ValueError("没有可用的说话人或 SFT 音色")


# ══════════════════════════════════════════════════════════════
#  应用生命周期
# ══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global cosyvoice_model, _infer_semaphore, SAMPLE_RATE

    _infer_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    for d in [SPEAKER_DIR, WAVS_DIR, LOGS_DIR, UPLOAD_DIR]:
        os.makedirs(d, exist_ok=True)

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required")

    # ── 加载模型 ──
    sys.path.insert(0, os.path.join(ROOT_DIR, "third_party", "Matcha-TTS"))
    sys.path.insert(0, ROOT_DIR)

    logger.info(f"加载模型: {MODEL_DIR} ...")
    t0 = time.time()

    load_jit = os.getenv("LOAD_JIT", "false").lower() in ("true", "1")
    load_trt = os.getenv("LOAD_TRT", "false").lower() in ("true", "1")
    load_vllm = os.getenv("LOAD_VLLM", "false").lower() in ("true", "1")
    fp16 = os.getenv("FP16", "false").lower() in ("true", "1")

    from cosyvoice.cli.cosyvoice import AutoModel
    cosyvoice_model = AutoModel(
        model_dir=MODEL_DIR,
        load_jit=load_jit, load_trt=load_trt,
        load_vllm=load_vllm, fp16=fp16,
    )
    SAMPLE_RATE = cosyvoice_model.sample_rate
    logger.info(f"模型加载完成 ({time.time()-t0:.1f}s), sr={SAMPLE_RATE}")

    # ── 检测模型能力 ──
    _detect_model_capabilities()

    # ── 注册说话人 (仅 zero_shot 模型) ──
    _register_all_speakers()

    # ── 预热 ──
    try:
        logger.info("预热推理 ...")
        if _registered_spk_ids:
            spk = sorted(_registered_spk_ids)[0]
            for _ in cosyvoice_model.inference_zero_shot("你好", "", "", zero_shot_spk_id=spk, stream=False):
                pass
        elif _model_caps["has_sft"] and _model_caps["sft_speakers"]:
            voice = _model_caps["sft_speakers"][0]
            for _ in cosyvoice_model.inference_sft("你好", voice, stream=False):
                pass
        logger.info("预热完成")
    except Exception as e:
        logger.warning(f"预热失败: {e}")

    yield

    logger.info("服务关闭")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="CosyVoice TTS API (全模型兼容)", version="3.0.0", lifespan=lifespan)
os.makedirs("static/wavs", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ══════════════════════════════════════════════════════════════
#  GET/POST /tts — 统一入口
# ══════════════════════════════════════════════════════════════

@app.api_route("/tts", methods=["GET", "POST"])
async def tts(request: Request):
    """
    统一 TTS 入口, 自动根据模型能力选择推理模式。

    参数:
      text (str, 必须):          要合成的文本
      voice (str, 默认"2222"):   说话人名称或 SFT 音色名
      type (str, 可选):          显式指定模式: "sft" / "zero_shot" / "instruct"
                                 为空则自动选择
      custom_voice (int):        >0 覆盖 voice
      speed (int, 默认5):        语速 1~9 (5=正常)
      wav (int, 默认0):          1=返回 WAV, 0=返回 JSON
      is_stream (int, 默认0):    1=流式返回 PCM
      text_seed (int, 默认42):   随机种子

      # 以下仅特定 type 使用:
      instruct_text (str):       (type=instruct) 指令文本
      prompt_text (str):         (type=zero_shot) 参考音频文本
      prompt_audio_path (str):   (type=zero_shot) 参考音频路径
    """
    params = await parse_request_params(request)

    text = get_param(params, "text", "", str).strip()
    if not text:
        return JSONResponse({"code": 1, "msg": "text params lost"})

    tts_type = get_param(params, "type", "", str).strip().lower()
    custom_voice = get_param(params, "custom_voice", 0, int)
    voice_raw = get_param(params, "voice", "2222", str)
    speed = get_param(params, "speed", 5, int)
    text_seed = get_param(params, "text_seed", 42, int)
    wav_flag = get_param(params, "wav", 0, int)
    is_stream = get_param(params, "is_stream", 0, int)
    instruct_text = get_param(params, "instruct_text", "", str)
    prompt_text = get_param(params, "prompt_text", "", str)
    prompt_audio_path = get_param(params, "prompt_audio_path", "", str)

    cosyvoice_speed = speed / 5.0 if speed > 0 else 1.0

    if text_seed > 0:
        torch.manual_seed(text_seed)

    logger.info(f"[/tts] type={tts_type!r} voice={voice_raw} text_len={len(text)} stream={is_stream}")

    # ── 显式指定 type ──
    if tts_type == "sft":
        if not _model_caps["has_sft"]:
            return JSONResponse({"code": 1, "msg": f"当前模型 ({_model_caps['model_type']}) 不支持 SFT"})
        sft_voice = voice_raw if voice_raw in _model_caps["sft_speakers"] else _model_caps["sft_speakers"][0]
        resolved = {"mode": "sft", "voice": sft_voice}

    elif tts_type == "zero_shot":
        if not _model_caps["has_zero_shot"]:
            return JSONResponse({"code": 1, "msg": f"当前模型 ({_model_caps['model_type']}) 不支持 zero_shot"})
        if prompt_text and prompt_audio_path and os.path.isfile(prompt_audio_path):
            # 实时 zero_shot (传参考音频)
            resolved = {"mode": "zero_shot_realtime", "prompt_text": prompt_text, "prompt_audio_path": prompt_audio_path}
        else:
            spk_id = voice_raw.replace(".pt", "").replace(".wav", "")
            if spk_id not in _registered_spk_ids:
                return JSONResponse({"code": 1, "msg": f"说话人 '{spk_id}' 未注册"})
            resolved = {"mode": "zero_shot_spk", "spk_id": spk_id}

    elif tts_type == "instruct":
        if not _model_caps["has_instruct"]:
            return JSONResponse({"code": 1, "msg": f"当前模型 ({_model_caps['model_type']}) 不支持 instruct"})
        if not instruct_text:
            return JSONResponse({"code": 1, "msg": "instruct 模式需要 instruct_text 参数"})
        resolved = {"mode": "instruct", "instruct_text": instruct_text, "voice": voice_raw}

    else:
        # ── 自动选择 ──
        resolved = resolve_voice(voice_raw, custom_voice)
        if resolved["mode"] == "none":
            return JSONResponse({"code": 1, "msg": "没有可用的说话人, 请注册说话人或使用 SFT 模型"})

    # ── 流式返回 ──
    if is_stream == 1:
        async def audio_stream():
            try:
                if resolved["mode"] == "instruct":
                    fn = getattr(cosyvoice_model, _model_caps["instruct_fn_name"])
                    gen = fn(text, resolved["voice"], resolved["instruct_text"], stream=True, speed=cosyvoice_speed)
                elif resolved["mode"] == "zero_shot_realtime":
                    gen = cosyvoice_model.inference_zero_shot(
                        text, resolved["prompt_text"], _load_wav_16k(resolved["prompt_audio_path"]),
                        stream=True, speed=cosyvoice_speed,
                    )
                else:
                    gen = _do_inference_generator(text, resolved, cosyvoice_speed)

                for item in gen:
                    sp = item["tts_speech"]
                    if sp.ndim == 1:
                        sp = sp.unsqueeze(0)
                    yield (sp.squeeze().numpy() * 32767).astype(np.int16).tobytes()
            except Exception as e:
                logger.error(f"流式推理错误: {e}", exc_info=True)

        return StreamingResponse(audio_stream(), media_type="audio/pcm",
                                 headers={"X-Sample-Rate": str(SAMPLE_RATE), "X-Channels": "1", "X-Bit-Depth": "16"})

    # ── 非流式 ──
    t0 = time.time()
    try:
        if resolved["mode"] == "instruct":
            fn = getattr(cosyvoice_model, _model_caps["instruct_fn_name"])
            results = await run_in_pool(lambda: list(fn(
                text, resolved["voice"], resolved["instruct_text"],
                stream=False, speed=cosyvoice_speed,
            )))
        elif resolved["mode"] == "zero_shot_realtime":
            p_speech = _load_wav_16k(resolved["prompt_audio_path"])
            results = await run_in_pool(lambda: list(cosyvoice_model.inference_zero_shot(
                text, resolved["prompt_text"], p_speech,
                stream=False, speed=cosyvoice_speed,
            )))
        else:
            results = await run_in_pool(lambda: _do_inference(text, resolved, cosyvoice_speed, False))
    except Exception as e:
        logger.error(f"推理错误: {e}", exc_info=True)
        return JSONResponse({"code": 1, "msg": str(e)})

    inference_time = round(time.time() - t0, 2)
    full_audio = collect_chunks(results)
    tag = f"use{inference_time}s-voice{voice_raw}-textlen{len(text)}"
    out_path, outname = save_to_wavs_dir(full_audio, SAMPLE_RATE, tag)

    try:
        audio_duration = round(sf.info(out_path).duration, 2)
    except Exception:
        audio_duration = round(full_audio.shape[1] / SAMPLE_RATE, 2)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if wav_flag > 0:
        return FileResponse(out_path, media_type="audio/x-wav")

    relative_url = f"/static/wavs/{outname}"
    host = request.headers.get("host", "localhost")
    return JSONResponse({
        "code": 0, "msg": "ok",
        "audio_files": [{"filename": out_path, "url": f"http://{host}{relative_url}",
                         "relative_url": relative_url, "inference_time": inference_time, "audio_duration": audio_duration}],
        "filename": out_path, "url": f"http://{host}{relative_url}", "relative_url": relative_url,
    })


# ══════════════════════════════════════════════════════════════
#  POST /tts/sft — SFT 预设音色 (仅 SFT 模型可用)
# ══════════════════════════════════════════════════════════════

@app.post("/tts/sft")
async def tts_sft(
    text: str = Form(..., description="要合成的文本"),
    voice: str = Form(default="中文女", description="SFT 预设音色名"),
    speed: float = Form(default=1.0, description="语速倍率"),
    stream: bool = Form(default=False),
    seed: int = Form(default=0),
):
    """SFT 预设音色合成 (需要 CosyVoice-300M-SFT / Instruct 模型)"""
    if not _model_caps["has_sft"]:
        return JSONResponse({"code": 1, "msg": f"当前模型 ({_model_caps['model_type']}) 不支持 SFT, "
                             f"请使用 CosyVoice-300M-SFT 或 CosyVoice-300M-Instruct"})
    if voice not in _model_caps["sft_speakers"]:
        return JSONResponse({"code": 1, "msg": f"音色 '{voice}' 不存在, 可用: {_model_caps['sft_speakers']}"})
    if seed > 0:
        torch.manual_seed(seed)

    t0 = time.time()
    results = await run_in_pool(lambda: list(cosyvoice_model.inference_sft(text, voice, stream=stream, speed=speed)))
    audio = collect_chunks(results)
    return StreamingResponse(io.BytesIO(audio_tensor_to_bytes(audio, SAMPLE_RATE)), media_type="audio/wav",
                             headers={"X-Inference-Time": str(round(time.time()-t0, 2)),
                                      "X-Audio-Duration": str(round(audio.shape[1]/SAMPLE_RATE, 2))})


# ══════════════════════════════════════════════════════════════
#  POST /tts/zero_shot — 上传参考音频克隆
# ══════════════════════════════════════════════════════════════

@app.post("/tts/zero_shot")
async def tts_zero_shot(
    text: str = Form(..., description="要合成的文本"),
    prompt_text: str = Form(..., description="参考音频中说的话"),
    prompt_audio: UploadFile = File(..., description="参考音频 (wav, >=16kHz, 3~10秒)"),
    speed: float = Form(default=1.0),
    stream: bool = Form(default=False),
    seed: int = Form(default=0),
    save_speaker: Optional[str] = Form(default=None, description="保存为说话人名称"),
):
    """上传参考音频实时克隆 (需要支持 zero_shot 的模型)"""
    if not _model_caps["has_zero_shot"]:
        return JSONResponse({"code": 1, "msg": f"当前模型 ({_model_caps['model_type']}) 不支持 zero_shot"})

    audio_bytes = await prompt_audio.read()
    tmp_path = os.path.join(UPLOAD_DIR, f"prompt_{uuid.uuid4().hex[:8]}.wav")
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)

    if seed > 0:
        torch.manual_seed(seed)

    t0 = time.time()
    try:
        prompt_speech_16k = _load_wav_16k(tmp_path)
        results = await run_in_pool(lambda: list(cosyvoice_model.inference_zero_shot(
            text, prompt_text, prompt_speech_16k, stream=stream, speed=speed)))
    finally:
        try: os.remove(tmp_path)
        except OSError: pass

    audio = collect_chunks(results)
    inference_time = round(time.time() - t0, 2)

    saved = None
    if save_speaker:
        spk_name = re.sub(r'[^\w\-]', '_', save_speaker.strip())
        if spk_name:
            torchaudio.save(os.path.join(SPEAKER_DIR, f"{spk_name}.wav"),
                            prompt_speech_16k.unsqueeze(0) if prompt_speech_16k.ndim == 1 else prompt_speech_16k, 16000)
            with open(os.path.join(SPEAKER_DIR, f"{spk_name}.txt"), "w", encoding="utf-8") as f:
                f.write(prompt_text)
            cosyvoice_model.add_zero_shot_spk(prompt_text, prompt_speech_16k, spk_name)
            _registered_spk_ids.add(spk_name)
            saved = f"{spk_name}.wav"

    headers = {"X-Inference-Time": str(inference_time), "X-Audio-Duration": str(round(audio.shape[1]/SAMPLE_RATE, 2))}
    if saved: headers["X-Saved-Speaker"] = saved
    return StreamingResponse(io.BytesIO(audio_tensor_to_bytes(audio, SAMPLE_RATE)), media_type="audio/wav", headers=headers)


# ══════════════════════════════════════════════════════════════
#  POST /tts/instruct — 自然语言指令控制
# ══════════════════════════════════════════════════════════════

@app.post("/tts/instruct")
async def tts_instruct(
    text: str = Form(..., description="要合成的文本"),
    voice: str = Form(default="中文女", description="基础音色 (SFT音色名 或 已注册说话人)"),
    instruct_text: str = Form(..., description="自然语言指令, 如 '用四川方言开心地说'"),
    speed: float = Form(default=1.0),
    stream: bool = Form(default=False),
    seed: int = Form(default=0),
):
    """指令控制合成 (需要 CosyVoice-300M-Instruct / CosyVoice2 / CosyVoice3)"""
    if not _model_caps["has_instruct"]:
        return JSONResponse({"code": 1, "msg": f"当前模型 ({_model_caps['model_type']}) 不支持 instruct"})
    if seed > 0:
        torch.manual_seed(seed)

    fn = getattr(cosyvoice_model, _model_caps["instruct_fn_name"])
    t0 = time.time()
    results = await run_in_pool(lambda: list(fn(text, voice, instruct_text, stream=stream, speed=speed)))
    audio = collect_chunks(results)
    return StreamingResponse(io.BytesIO(audio_tensor_to_bytes(audio, SAMPLE_RATE)), media_type="audio/wav",
                             headers={"X-Inference-Time": str(round(time.time()-t0, 2)),
                                      "X-Audio-Duration": str(round(audio.shape[1]/SAMPLE_RATE, 2))})


# ══════════════════════════════════════════════════════════════
#  POST /tts/stream — 流式合成
# ══════════════════════════════════════════════════════════════

@app.post("/tts/stream")
async def tts_stream(
    text: str = Form(...),
    voice: str = Form(default=""),
    prompt_text: Optional[str] = Form(default=None),
    prompt_audio: Optional[UploadFile] = File(default=None),
    instruct_text: Optional[str] = Form(default=None),
    speed: float = Form(default=1.0),
    seed: int = Form(default=0),
):
    """流式合成, 根据参数自动选择 sft/zero_shot/instruct 模式"""

    async def audio_stream():
        if seed > 0:
            torch.manual_seed(seed)
        prompt_audio_path = None

        try:
            # instruct 模式
            if instruct_text and _model_caps["has_instruct"]:
                fn = getattr(cosyvoice_model, _model_caps["instruct_fn_name"])
                gen = fn(text, voice, instruct_text, stream=True, speed=speed)

            # 实时 zero_shot (上传音频)
            elif prompt_audio and prompt_text and _model_caps["has_zero_shot"]:
                ab = await prompt_audio.read()
                prompt_audio_path = os.path.join(UPLOAD_DIR, f"stream_{uuid.uuid4().hex[:8]}.wav")
                with open(prompt_audio_path, "wb") as f:
                    f.write(ab)
                gen = cosyvoice_model.inference_zero_shot(
                    text, prompt_text, _load_wav_16k(prompt_audio_path), stream=True, speed=speed)

            # 已注册说话人 / SFT
            else:
                resolved = resolve_voice(voice)
                if resolved["mode"] == "none":
                    return
                gen = _do_inference_generator(text, resolved, speed)

            for item in gen:
                sp = item["tts_speech"]
                if sp.ndim == 1:
                    sp = sp.unsqueeze(0)
                yield (sp.squeeze().numpy() * 32767).astype(np.int16).tobytes()
        finally:
            if prompt_audio_path:
                try: os.remove(prompt_audio_path)
                except OSError: pass

    return StreamingResponse(audio_stream(), media_type="audio/pcm",
                             headers={"X-Sample-Rate": str(SAMPLE_RATE), "X-Channels": "1", "X-Bit-Depth": "16"})


# ══════════════════════════════════════════════════════════════
#  /speakers, /clear_wavs, /health
# ══════════════════════════════════════════════════════════════

@app.get("/speakers")
async def list_speakers():
    custom = sorted([f for f in os.listdir(SPEAKER_DIR) if os.path.splitext(f)[1].lower() in _AUDIO_EXTS]) if os.path.isdir(SPEAKER_DIR) else []
    return JSONResponse({"code": 0, "msg": "ok", "speakers": custom,
                         "registered": sorted(_registered_spk_ids),
                         "sft_speakers": _model_caps["sft_speakers"]})


@app.post("/speakers/upload")
async def upload_speaker(
    name: str = Form(...), audio: UploadFile = File(...), prompt_text: str = Form(default=""),
):
    if not _model_caps["has_zero_shot"]:
        return JSONResponse({"code": 1, "msg": "当前模型不支持 zero_shot, 无法注册说话人"})
    spk_name = re.sub(r'[^\w\-]', '_', name.strip())
    if not spk_name:
        return JSONResponse({"code": 1, "msg": "Invalid speaker name"})

    audio_bytes = await audio.read()
    wav_path = os.path.join(SPEAKER_DIR, f"{spk_name}.wav")
    with open(wav_path, "wb") as f:
        f.write(audio_bytes)
    if prompt_text:
        with open(os.path.join(SPEAKER_DIR, f"{spk_name}.txt"), "w", encoding="utf-8") as f:
            f.write(prompt_text)

    try:
        _register_audio_speaker(spk_name, wav_path)
    except Exception as e:
        return JSONResponse({"code": 1, "msg": str(e)})
    return JSONResponse({"code": 0, "msg": "ok", "speaker": f"{spk_name}.wav"})


@app.delete("/speakers/{name}")
async def delete_speaker(name: str):
    bare = os.path.basename(name)
    for ext in (*_AUDIO_EXTS, ".txt"):
        bare = bare.replace(ext, "")
    deleted = []
    for ext in (*_AUDIO_EXTS, ".txt"):
        fp = os.path.join(SPEAKER_DIR, f"{bare}{ext}")
        if os.path.isfile(fp):
            os.remove(fp)
            deleted.append(f"{bare}{ext}")
    if not deleted:
        return JSONResponse({"code": 1, "msg": f"说话人 '{bare}' 不存在"}, status_code=404)
    _registered_spk_ids.discard(bare)
    return JSONResponse({"code": 0, "msg": "ok", "deleted": deleted})


@app.post("/clear_wavs")
async def clear_wavs():
    count = 0
    for f in os.listdir(WAVS_DIR):
        fp = os.path.join(WAVS_DIR, f)
        if os.path.isfile(fp):
            os.remove(fp)
            count += 1
    return JSONResponse({"code": 0, "msg": f"Cleared {count} files"})


@app.get("/health")
async def health():
    gpu = {}
    if torch.cuda.is_available():
        gpu = {"gpu_name": torch.cuda.get_device_name(0),
               "gpu_memory_total_mb": round(torch.cuda.get_device_properties(0).total_mem / 1024**2),
               "gpu_memory_used_mb": round(torch.cuda.memory_allocated(0) / 1024**2)}
    return JSONResponse({
        "code": 0, "msg": "ok", "status": "healthy",
        "model_dir": MODEL_DIR, "model_type": _model_caps["model_type"],
        "capabilities": {k: v for k, v in _model_caps.items() if k != "instruct_fn_name"},
        "sample_rate": SAMPLE_RATE, "max_concurrency": MAX_CONCURRENCY,
        "registered_speakers": sorted(_registered_spk_ids), "gpu": gpu,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_all_model:app", host=HOST, port=PORT, workers=1, log_level="info", timeout_keep_alive=300)
