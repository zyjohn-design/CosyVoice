# CosyVoice 生产部署指南 (GPU & NPU)

> 本文档是项目根 `README.md` 的补充，介绍 **NVIDIA GPU** 和 **华为昇腾 NPU 910B** 的完整 Docker 部署方法、挂载清单、环境变量、性能基线与故障排查。

## 📚 docs/ 目录其他文档

| 文档 | 内容 |
| :--- | :--- |
| [`NPU_ADAPT.md`](./NPU_ADAPT.md) | NPU 910B 适配技术细节 (CANN / torch_npu / transfer_to_npu / 自研优化历史) |
| [`TECH_DOC_V1.md`](./TECH_DOC_V1.md) | CosyVoice 引擎技术文档 (架构 / 模块 / 训练) |
| [`VOICE_TTS_API_DOC.md`](./VOICE_TTS_API_DOC.md) | TTS API 接口文档 (路由 / 参数 / 响应格式) |

---


本仓库提供了**两套生产级 Docker 部署方案**，针对不同硬件：

| 方案 | 硬件 | 推理引擎 | 主入口 | 实测 RTF (Chinese, 14-42字) |
| :--- | :--- | :--- | :--- | :--- |
| **GPU 版** | NVIDIA L20 / A800 | PyTorch + (可选)TorchScript JIT | `app.py` | 0.3-0.5 |
| **NPU 版** | 华为昇腾 910B | vLLM-Ascend (v5 默认) | `app_npu.py` | **0.56-0.72** |

两套方案都通过 FastAPI 暴露**统一的 TTS API**（`/tts`、`/tts/zero_shot`、`/tts/stream`、`/speakers/*`、`/health`），客户端代码完全通用。

---

## 📋 通用前置准备 (Pre-flight Checklist)

无论 GPU 还是 NPU 部署，以下步骤都必须先完成：

### 1️⃣ 克隆仓库并初始化子模块

```bash
git clone <this-repo-url> CosyVoice
cd CosyVoice
git submodule update --init --recursive    # 拉取 Matcha-TTS 子模块（必须！）
```

> ⚠️ **重要**：`third_party/Matcha-TTS` 是 git submodule，缺失会导致 import 报错。

### 2️⃣ 下载预训练模型 (推荐 modelscope 加速)

```bash
# 用 modelscope CLI（推荐，国内快）
pip install modelscope
mkdir -p pretrained_models
modelscope download --model iic/CosyVoice2-0.5B --local_dir pretrained_models/CosyVoice2-0.5B

# 或者用 huggingface
huggingface-cli download FunAudioLLM/CosyVoice2-0.5B --local-dir pretrained_models/CosyVoice2-0.5B
```

模型目录结构（成功后应看到）：
```
pretrained_models/CosyVoice2-0.5B/
├── cosyvoice2.yaml                 # 模型配置（AutoModel 据此判定 V2）
├── llm.pt                          # LLM 权重 (~1.0 GB)
├── flow.pt                         # Flow matching 权重
├── flow.cache.pt                   # Flow 缓存
├── hift.pt                         # HiFi-GAN 声码器权重
├── campplus.onnx                   # 说话人嵌入
├── speech_tokenizer_v2.onnx        # 语音分词器
├── spk2info.pt                     # SFT 说话人字典（V2 可空）
├── CosyVoice-BlankEN/              # 文本前端
└── ...
```

### 3️⃣ 准备 wetext 缓存（文本归一化，离线运行）

`app.py` / `app_npu.py` 用 monkey-patch 让 wetext 走本地缓存，避免运行时联网下载。

```bash
# 创建宿主机缓存目录（任何位置都可以，下面以容器内默认路径为例）
mkdir -p ~/.cache/modelscope/hub/pengzhendong/wetext

# 提前下载 wetext (一次性，约 800MB)
modelscope download --model pengzhendong/wetext \
  --local_dir ~/.cache/modelscope/hub/pengzhendong/wetext
```

### 4️⃣ 准备说话人音频 (用于 zero-shot 克隆)

在仓库根目录 `speakers/` 放置参考音频和对应文本：

```bash
mkdir -p speakers
# 格式: {名字}.wav (3-10秒, ≥16kHz) + {名字}.txt (该音频的转写文本)
# 例:
speakers/
├── 1111-男声.wav
├── 1111-男声.txt
├── 10003-女声.wav
├── 10003-女声.txt
├── 10004-女声.wav
└── 10004-女声.txt
```

服务启动时会自动扫描 `speakers/` 并注册到模型。注册成功后通过 API 用 `name=10003-女声` 直接调用。

---

## 🟢 GPU 部署 (NVIDIA L20 / A800 / 4090 / H100)

### 镜像构建

```bash
# 基于 nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04，含 conda + Python 3.10
docker compose build           # 用 Dockerfile + docker-compose.yml
```

构建用时约 15-30 分钟（取决于网络）。镜像名 `cosyvoice-tts-api:latest`，约 18 GB。

### docker-compose.yml 关键挂载

| 宿主机路径 | 容器路径 | 作用 |
| :--- | :--- | :--- |
| `${MODEL_HOST_DIR:-./pretrained_models}` | `/workspace/CosyVoice/pretrained_models` | 模型权重（**必挂**） |
| `${WETEXT_CACHE_DIR}` | `/root/.cache/modelscope/hub/pengzhendong/wetext` | wetext 缓存（**必挂**，避免联网） |
| `./speakers` | `/workspace/CosyVoice/speakers` | 说话人音频 |
| `./static/wavs` | `/workspace/CosyVoice/static/wavs` | 生成音频输出 |
| `./logs` | `/workspace/CosyVoice/logs` | 服务日志 |

**核心环境变量** (docker-compose.yml `environment:` 段)：

| 变量 | 默认 | 说明 |
| :--- | :--- | :--- |
| `MODEL_DIR` | `pretrained_models/CosyVoice2-0.5B` | 模型相对路径 |
| `MAX_CONCURRENCY` | `4` | L20 建议 4-6，A800 建议 8-12 |
| `LOAD_JIT` | `true` | TorchScript 加速（推荐开） |
| `LOAD_TRT` | `false` | TensorRT 加速（需先构建 `.plan` 引擎） |
| `LOAD_VLLM` | `false` | vLLM 加速 LLM（需额外装 vllm 包） |
| `FP16` | `true` | 半精度推理（推荐开） |
| `GPU_DEVICE_ID` | `0` | 用第几张 GPU |
| `API_PORT` | `8080` | 宿主机端口 |

### GPU 启动命令

```bash
# 第 1 张 GPU、默认端口 8080
docker compose up -d

# 自定义：用第 2 张 GPU、端口 8090、并发 8
GPU_DEVICE_ID=1 API_PORT=8090 MAX_CONCURRENCY=8 docker compose up -d

# 查看日志、等待预热完成
docker compose logs -f cosyvoice-tts
```

启动后 60-180 秒（取决于 GPU），日志看到 `预热全部完成, 服务就绪` 即可调用。

### GPU 自检

```bash
curl http://localhost:8080/health
curl "http://localhost:8080/tts?text=你好世界&name=10003-女声" --output test.wav
```

### GPU 性能基线 (NVIDIA A800)

> 配置: `LOAD_JIT=true LOAD_TRT=true LOAD_VLLM=true FP16=true`，说话人 `1111-男声`，单并发同步请求，10 条递增长度文本。

```text
────────────────────────────────────────────────────────────────────────────────────
  CosyVoice TTS 性能测试报告 — NVIDIA A800
  说话人: 1111-男声
  时间: 2026-03-27 13:33:48
────────────────────────────────────────────────────────────────────────────────────
  #  描述         字数    推理(s)   音频(s)   客户端(s)    RTF    字/秒   状态
────────────────────────────────────────────────────────────────────────────────────
  1  5字           5     0.22     0.92      0.22     0.239   22.7   OK
  2  10字         12     0.47     3.28      0.49     0.143   25.5   OK
  3  15字         18     0.72     4.80      0.74     0.150   25.0   OK
  4  20字         27     0.77     5.44      0.79     0.142   35.1   OK
  5  25字         32     1.07     7.40      1.08     0.145   29.9   OK
  6  30字         39     1.15     8.32      1.16     0.138   33.9   OK
  7  35字         47     1.12     9.20      1.13     0.122   42.0   OK
  8  40字         53     1.22    10.28      1.24     0.119   43.4   OK
  9  45字         60     1.81    13.52      1.83     0.134   33.1   OK
 10  50字         62     1.41    11.92      1.43     0.118   44.0   OK
────────────────────────────────────────────────────────────────────────────────────
 合计           355     9.96    75.08              0.133    35.6   10/10 OK
────────────────────────────────────────────────────────────────────────────────────
```

**摘要**: 平均 RTF **0.133**（7.5× 实时），平均吞吐 **35.6 字/秒**，全部成功。

---

## 🟣 NPU 部署 (华为昇腾 910B) ★

> 实测 RTF **0.56-0.72**（Chinese, vLLM-Ascend 0.17 + CosyVoice2-0.5B + 910B 单卡）。

### NPU 镜像构建

基础镜像 `quay.io/ascend/vllm-ascend:v0.17.0rc1`（含 CANN 8.x + torch_npu + PyTorch + vLLM-Ascend）。

```bash
docker compose -f docker-compose.npu.yml build
```

如果内网受限，可用本地 pip 源加速：
```bash
docker compose -f docker-compose.npu.yml build \
  --build-arg PIP_INDEX_URL=http://your-mirror/simple \
  --build-arg PIP_TRUSTED_HOST=your-mirror
```

镜像约 20 GB。

### docker-compose.npu.yml 关键挂载

NPU 部署有**两类**挂载：宿主机 NPU 驱动（必须，否则 NPU 无法访问）+ 项目文件。

#### A. 宿主机 NPU 驱动挂载（**必须，全部 6 条**）

```yaml
- /usr/local/dcmi:/usr/local/dcmi
- /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool
- /usr/local/bin/npu-smi:/usr/local/bin/npu-smi
- /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/
- /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info
- /etc/ascend_install.info:/etc/ascend_install.info
```

> 这 6 条让容器能调用宿主机的 NPU 驱动 + 工具。**任何一条缺失都会导致 NPU 探测失败。**

#### B. NPU 设备文件映射（**必须**）

```yaml
devices:
  - /dev/davinci${NPU_DEVICE_ID:-0}:/dev/davinci${NPU_DEVICE_ID:-0}    # 用哪张卡
  - /dev/davinci_manager:/dev/davinci_manager
  - /dev/devmm_svm:/dev/devmm_svm
  - /dev/hisi_hdc:/dev/hisi_hdc
```

#### C. 项目挂载

| 宿主机路径 | 容器路径 | 作用 |
| :--- | :--- | :--- |
| `${MODEL_HOST_DIR:-./pretrained_models}` | `/workspace/CosyVoice/pretrained_models` | 模型权重（**必挂**） |
| `${WETEXT_CACHE_DIR:-./pengzhendong/wetext}` | `/root/.cache/modelscope/hub/pengzhendong/wetext` | wetext 缓存（**必挂**） |
| `./Matcha-TTS` | `/workspace/CosyVoice/third_party/Matcha-TTS` | submodule 热挂载 |
| `./app_npu.py` | `/workspace/CosyVoice/app_npu.py` | NPU 版 API 入口（热更新） |
| `./npu_optimizer.py` | `/workspace/CosyVoice/npu_optimizer.py` | NPU 优化模块（仅 fallback 路径生效） |
| `./cosyvoice2_npu.py` | `/workspace/CosyVoice/cosyvoice/vllm/cosyvoice2.py` | vLLM 模型注册（**v5 关键**） |
| `./scripts` | `/workspace/CosyVoice/scripts` | vLLM 冒烟测试脚本 |
| `./logs` | `/workspace/CosyVoice/logs` | 日志 |

> ⚠️ **注意**：`./speakers:/workspace/CosyVoice/speakers` 默认**已注释**。说话人音频在构建镜像时已 `COPY` 进容器。如要热挂载，去掉 `# - ./speakers:...` 前的注释。

### 核心环境变量 (NPU 专属)

| 变量 | 默认 | 说明 |
| :--- | :--- | :--- |
| `LOAD_VLLM` | **`true`** ★ | **v5 主推 vLLM-Ascend**，无早停 bug |
| `NPU_DEVICE_ID` | `0` | 用第几张 NPU 卡 (`/dev/davinci0`) |
| `ASCEND_RT_VISIBLE_DEVICES` | `0` | CANN runtime 卡可见性，**必须** = `NPU_DEVICE_ID` |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | NPU 多进程必须用 spawn（fork 会崩） |
| `VLLM_USE_V1` | `1` | 强制 vLLM V1 引擎 |
| `FP16` | `true` | NPU 910B 推荐 FP16 |
| `MAX_CONCURRENCY` | `4` | NPU 910B 单卡建议 4-8 |

#### Flow 优化（vLLM 模式下不生效，fallback 模式生效）

| 变量 | 默认 | 说明 |
| :--- | :--- | :--- |
| `NPU_FLOW_TIMESTEPS` | `4` | Flow ODE 步数（10→4 砍 60%） |
| `NPU_DISABLE_CFG` | `true` | CFG-free（batch 2→1） |

#### Fallback 路径专用（**默认全关**，仅在 `LOAD_VLLM=false` 时生效）

| 变量 | 默认 | 说明 |
| :--- | :--- | :--- |
| `NPU_FUSED_ATTN` | `false` | ⚠️ 自研 fused attention（**有早停 bug，不要开**） |
| `NPU_PROFILE` | `false` | 模块级耗时诊断 |

### NPU 启动命令

```bash
# 推荐：默认 vLLM-Ascend 路径
docker compose -f docker-compose.npu.yml up -d

# 自定义：用 2 号 NPU 卡
NPU_DEVICE_ID=2 docker compose -f docker-compose.npu.yml up -d

# 查看启动日志（首次 vLLM 模型导出耗时 60-120s）
docker compose -f docker-compose.npu.yml logs -f cosyvoice-tts
```

预期看到：
```
模型加载完成 (133.3s), 类型=CosyVoice2, 参数={load_vllm=True, fp16=True}
说话人注册完成: 3/3
预热全部完成, 服务就绪
Uvicorn running on http://0.0.0.0:8080
```

### NPU 冒烟自检 (强烈推荐)

容器跑起来后，跑一次冒烟脚本验证 vLLM 全链路：

```bash
docker exec -it cosyvoice-tts \
  python /workspace/CosyVoice/scripts/smoke_vllm_npu.py
```

会按 6 步检查：torch_npu / vllm import / vllm-ascend backend / CosyVoice2 注册 / 模型目录 / EngineArgs。全绿就放心。

### NPU 性能基线 (v5 vLLM-Ascend, 910B 单卡)

> 配置: `LOAD_VLLM=true FP16=true`，说话人 `1111-男声`，单并发同步请求，10 条递增长度文本。

```text
────────────────────────────────────────────────────────────────────────────────────
  CosyVoice TTS 性能测试报告 — 华为昇腾 910B (vLLM-Ascend)
  说话人: 1111-男声
  时间: 2026-05-28 13:34:37
────────────────────────────────────────────────────────────────────────────────────
  #  描述         字数    推理(s)   音频(s)   客户端(s)    RTF    字/秒   状态
────────────────────────────────────────────────────────────────────────────────────
  1  5字           5     1.48     0.92      1.49     1.609    3.4   OK
  2  10字         12     2.62     3.32      2.63     0.789    4.6   OK
  3  15字         18     3.17     4.72      3.18     0.672    5.7   OK
  4  20字         27     4.21     6.92      4.21     0.608    6.4   OK
  5  25字         32     4.59     7.96      4.59     0.577    7.0   OK
  6  30字         39     4.61     8.24      4.62     0.559    8.5   OK
  7  35字         47     5.68    10.08      5.69     0.563    8.3   OK
  8  40字         53     7.19    13.48      7.20     0.533    7.4   OK
  9  45字         60     6.73    12.12      6.74     0.555    8.9   OK
 10  50字         62     6.72    12.08      6.73     0.556    9.2   OK
────────────────────────────────────────────────────────────────────────────────────
 合计           355    47.00    79.84              0.589    7.6   10/10 OK
────────────────────────────────────────────────────────────────────────────────────
```

**摘要**: 平均 RTF **0.589**（1.7× 实时），平均吞吐 **7.6 字/秒**，全部成功，音频长度完全正常（v3/v4 的早停 bug 彻底消失）。

> 对比 v3/v4 自研 fused attention：speech_len 仅为正常的 24-50%（早停 bug）。**v5 切到 vLLM 后早停彻底消失，质量无损。**

### 🆚 GPU vs NPU 性能对比 (单并发, 同一组 10 条文本, 总计 355 字)

| 指标 | NVIDIA A800 | 华为昇腾 910B | 倍数 |
| :--- | :---: | :---: | :---: |
| 总推理时间 | **9.96 s** | 47.00 s | A800 快 **4.7×** |
| 平均 RTF | **0.133** | 0.589 | A800 低 **4.4×** |
| 平均吞吐 | **35.6 字/秒** | 7.6 字/秒 | A800 高 **4.7×** |
| 短文本首字延迟 (5字) | **0.22 s** | 1.48 s | A800 低 **6.7×** |
| 长文本生成 (50字) | **1.41 s** | 6.72 s | A800 低 **4.8×** |
| 音频长度一致性 | ✓ (与目标对齐) | ✓ (与目标对齐) | 相同 |
| 成功率 | 10/10 | 10/10 | 相同 |

**判定**：
- A800 在所有指标上稳定领先 NPU 910B 约 4-5×（得益于 TRT/JIT/vLLM 全开 + CUDA 生态成熟度）
- NPU 910B RTF=0.59 已满足**生产实时合成**需求（1.7× 实时），且**质量与 A800 完全一致**
- 短文本（5字）NPU 有明显冷启动开销（RTF=1.6），建议用预热避免影响首请求体感

> 测试脚本: `test_tts.py`。复现命令: `python test_tts.py --host <host> --port 8080 --speaker 1111-男声`

### NPU Fallback 路径（vLLM 跑不通时）

```bash
# 关掉 vLLM，回退到 v4 npu_optimizer（仅 Flow 优化有效）
LOAD_VLLM=false docker compose -f docker-compose.npu.yml up -d
```

⚠️ 即便回退，**也不要打开 `NPU_FUSED_ATTN=true`** — 已实证有早停 bug。

---

## 🔌 客户端调用示例

无论 GPU 还是 NPU 部署，API 完全一致：

```bash
# 同步生成 (返回 wav)
curl "http://localhost:8080/tts?text=你好世界&name=10003-女声" --output out.wav

# 列出说话人
curl http://localhost:8080/speakers

# 实时上传参考音频做 zero-shot 克隆
curl -X POST http://localhost:8080/tts/zero_shot \
  -F "text=要合成的文字" \
  -F "prompt_text=参考音频里说的话" \
  -F "prompt_audio=@reference.wav" \
  --output cloned.wav

# 流式合成（chunk-by-chunk 返回 PCM）
curl -N "http://localhost:8080/tts/stream?text=很长很长的文本&name=10003-女声" --output stream.pcm

# 健康检查（含 GPU/NPU 统计）
curl http://localhost:8080/health
```

或用 Python:
```python
import requests
r = requests.get("http://localhost:8080/tts", params={"text": "你好", "name": "10003-女声"})
open("out.wav", "wb").write(r.content)
```

性能测试：
```bash
python test_tts.py --host 127.0.0.1 --port 8080
```

---

## 🛠️ 常见故障排查

| 现象 | 原因 | 解决 |
| :--- | :--- | :--- |
| `Matcha-TTS` import 报错 | git submodule 未拉 | `git submodule update --init --recursive` |
| wetext 启动卡在下载 | 缺 wetext 缓存挂载 | 提前下载并挂载 `${WETEXT_CACHE_DIR}` |
| GPU 容器起不来 | nvidia-container-toolkit 未装 | `apt install nvidia-container-toolkit && systemctl restart docker` |
| NPU `torch.npu.is_available()=False` | 6 条驱动挂载缺一 / `ASCEND_RT_VISIBLE_DEVICES` 错 | 跑 `scripts/smoke_vllm_npu.py` 定位 |
| NPU 推理音频太短（早停 bug） | 走了 fallback 且开了 `NPU_FUSED_ATTN` | 关掉 `NPU_FUSED_ATTN=false`，确保 `LOAD_VLLM=true` |
| vLLM 模型导出失败 | 模型目录权限/磁盘空间 | 确保 `pretrained_models/CosyVoice2-0.5B/` 可写、有 2GB 空间放 `vllm/` 子目录 |
