![SVG Banners](https://svg-banners.vercel.app/api?type=origin&text1=CosyVoice🤠&text2=Text-to-Speech%20💖%20Large%20Language%20Model&width=800&height=210)

## 👉🏻 CosyVoice 👈🏻

**Fun-CosyVoice 3.0**: [Demos](https://funaudiollm.github.io/cosyvoice3/); [Paper](https://arxiv.org/pdf/2505.17589); [Modelscope](https://www.modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512); [Huggingface](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512); [CV3-Eval](https://github.com/FunAudioLLM/CV3-Eval)

**CosyVoice 2.0**: [Demos](https://funaudiollm.github.io/cosyvoice2/); [Paper](https://arxiv.org/pdf/2412.10117); [Modelscope](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B); [HuggingFace](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B)

**CosyVoice 1.0**: [Demos](https://fun-audio-llm.github.io); [Paper](https://funaudiollm.github.io/pdf/CosyVoice_v1.pdf); [Modelscope](https://www.modelscope.cn/models/iic/CosyVoice-300M); [HuggingFace](https://huggingface.co/FunAudioLLM/CosyVoice-300M)

## Highlight🔥

**Fun-CosyVoice 3.0** is an advanced text-to-speech (TTS) system based on large language models (LLM), surpassing its predecessor (CosyVoice 2.0) in content consistency, speaker similarity, and prosody naturalness. It is designed for zero-shot multilingual speech synthesis in the wild.
### Key Features
- **Language Coverage**: Covers 9 common languages (Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian), 18+ Chinese dialects/accents (Guangdong, Minnan, Sichuan, Dongbei, Shan3xi, Shan1xi, Shanghai, Tianjin, Shandong, Ningxia, Gansu, etc.) and meanwhile supports both multi-lingual/cross-lingual zero-shot voice cloning.
- **Content Consistency & Naturalness**: Achieves state-of-the-art performance in content consistency, speaker similarity, and prosody naturalness.
- **Pronunciation Inpainting**: Supports pronunciation inpainting of Chinese Pinyin and English CMU phonemes, providing more controllability and thus suitable for production use.
- **Text Normalization**: Supports reading of numbers, special symbols and various text formats without a traditional frontend module.
- **Bi-Streaming**: Support both text-in streaming and audio-out streaming, and achieves latency as low as 150ms while maintaining high-quality audio output.
- **Instruct Support**: Supports various instructions such as languages, dialects, emotions, speed, volume, etc.


## Roadmap

- [x] 2025/12

    - [x] release Fun-CosyVoice3-0.5B-2512 base model, rl model and its training/inference script
    - [x] release Fun-CosyVoice3-0.5B modelscope gradio space

- [x] 2025/08

    - [x] Thanks to the contribution from NVIDIA Yuekai Zhang, add triton trtllm runtime support and cosyvoice2 grpo training support

- [x] 2025/07

    - [x] release Fun-CosyVoice 3.0 eval set

- [x] 2025/05

    - [x] add CosyVoice2-0.5B vllm support

- [x] 2024/12

    - [x] 25hz CosyVoice2-0.5B released

- [x] 2024/09

    - [x] 25hz CosyVoice-300M base model
    - [x] 25hz CosyVoice-300M voice conversion function

- [x] 2024/08

    - [x] Repetition Aware Sampling(RAS) inference for llm stability
    - [x] Streaming inference mode support, including kv cache and sdpa for rtf optimization

- [x] 2024/07

    - [x] Flow matching training support
    - [x] WeTextProcessing support when ttsfrd is not available
    - [x] Fastapi server and client

## Evaluation

| Model | Open-Source | Model Size | test-zh<br>CER (%) ↓ | test-zh<br>SS (%) ↑ | test-en<br>WER (%) ↓ | test-en<br>SS (%) ↑ | test-hard<br>CER (%) ↓ | test-hard<br>SS (%) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Human | - | - | 1.26 | 75.5 | 2.14 | 73.4 | - | - |
| Seed-TTS | ❌ | - | 1.12 | 79.6 | 2.25 | 76.2 | 7.59 | 77.6 |
| MiniMax-Speech | ❌ | - | 0.83 | 78.3 | 1.65 | 69.2 | - | - |
| F5-TTS | ✅ | 0.3B | 1.52 | 74.1 | 2.00 | 64.7 | 8.67 | 71.3 |
| Spark TTS | ✅ | 0.5B | 1.2 | 66.0 | 1.98 | 57.3 | - | - |
| CosyVoice2 | ✅ | 0.5B | 1.45 | 75.7 | 2.57 | 65.9 | 6.83 | 72.4 |
| FireRedTTS2 | ✅ | 1.5B | 1.14 | 73.2 | 1.95 | 66.5 | - | - |
| Index-TTS2 | ✅ | 1.5B | 1.03 | 76.5 | 2.23 | 70.6 | 7.12 | 75.5 |
| VibeVoice-1.5B | ✅ | 1.5B | 1.16 | 74.4 | 3.04 | 68.9 | - | - |
| VibeVoice-Realtime | ✅ | 0.5B | - | - | 2.05 | 63.3 | - | - |
| HiggsAudio-v2 | ✅ | 3B | 1.50 | 74.0 | 2.44 | 67.7 | - | - |
| VoxCPM | ✅ | 0.5B | 0.93 | 77.2 | 1.85 | 72.9 | 8.87 | 73.0 |
| GLM-TTS | ✅ | 1.5B | 1.03 | 76.1 | - | - | - | - |
| GLM-TTS RL | ✅ | 1.5B | 0.89 | 76.4 | - | - | - | - |
| Fun-CosyVoice3-0.5B-2512 | ✅ | 0.5B | 1.21 | 78.0 | 2.24 | 71.8 | 6.71 | 75.8 |
| Fun-CosyVoice3-0.5B-2512_RL | ✅ | 0.5B | 0.81 | 77.4 | 1.68 | 69.5 | 5.44 | 75.0 |


## Install

### Clone and install

- Clone the repo
    ``` sh
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
    # If you failed to clone the submodule due to network failures, please run the following command until success
    cd CosyVoice
    git submodule update --init --recursive
    ```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:

    ``` sh
    conda create -n cosyvoice -y python=3.10
    conda activate cosyvoice
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

    # If you encounter sox compatibility issues
    # ubuntu
    sudo apt-get install sox libsox-dev
    # centos
    sudo yum install sox sox-devel
    ```

### Model download

We strongly recommend that you download our pretrained `Fun-CosyVoice3-0.5B` `CosyVoice2-0.5B` `CosyVoice-300M` `CosyVoice-300M-SFT` `CosyVoice-300M-Instruct` model and `CosyVoice-ttsfrd` resource.

``` python
# modelscope SDK model download
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='pretrained_models/CosyVoice-300M-Instruct')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')

# for oversea users, huggingface SDK model download
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('FunAudioLLM/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
snapshot_download('FunAudioLLM/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
snapshot_download('FunAudioLLM/CosyVoice-300M-Instruct', local_dir='pretrained_models/CosyVoice-300M-Instruct')
snapshot_download('FunAudioLLM/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```

Optionally, you can unzip `ttsfrd` resource and install `ttsfrd` package for better text normalization performance.

Notice that this step is not necessary. If you do not install `ttsfrd` package, we will use wetext by default.

``` sh
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```

### Basic Usage

We strongly recommend using `Fun-CosyVoice3-0.5B` for better performance.
Follow the code in `example.py` for detailed usage of each model.
```sh
python example.py
```

#### vLLM Usage
CosyVoice2/3 now supports **vLLM 0.11.x+ (V1 engine)** and **vLLM 0.9.0 (legacy)**.
Older vllm version(<0.9.0) do not support CosyVoice inference, and versions in between (e.g., 0.10.x) are not tested.

Notice that `vllm` has a lot of specific requirements. You can create a new env to in case your hardward do not support vllm and old env is corrupted.

``` sh
conda create -n cosyvoice_vllm --clone cosyvoice
conda activate cosyvoice_vllm
# for vllm==0.9.0
pip install vllm==v0.9.0 transformers==4.51.3 numpy==1.26.4 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
# for vllm>=0.11.0
pip install vllm==v0.11.0 transformers==4.57.1 numpy==1.26.4 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
python vllm_example.py
```

#### Start web demo

You can use our web demo page to get familiar with CosyVoice quickly.

Please see the demo website for details.

``` python
# change iic/CosyVoice-300M-SFT for sft inference, or iic/CosyVoice-300M-Instruct for instruct inference
python3 webui.py --port 50000 --model_dir pretrained_models/CosyVoice-300M
```

#### Advanced Usage

For advanced users, we have provided training and inference scripts in `examples/libritts`.

#### Build for deployment

Optionally, if you want service deployment,
You can run the following steps.

``` sh
cd runtime/python
docker build -t cosyvoice:v1.0 .
# change iic/CosyVoice-300M to iic/CosyVoice-300M-Instruct if you want to use instruct inference
# for grpc usage
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/grpc && python3 server.py --port 50000 --max_conc 4 --model_dir iic/CosyVoice-300M && sleep infinity"
cd grpc && python3 client.py --port 50000 --mode <sft|zero_shot|cross_lingual|instruct>
# for fastapi usage
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/fastapi && python3 server.py --port 50000 --model_dir iic/CosyVoice-300M && sleep infinity"
cd fastapi && python3 client.py --port 50000 --mode <sft|zero_shot|cross_lingual|instruct>
```

#### Using Nvidia TensorRT-LLM for deployment

Using TensorRT-LLM to accelerate cosyvoice2 llm could give 4x acceleration comparing with huggingface transformers implementation.
To quick start:

``` sh
cd runtime/triton_trtllm
docker compose up -d
```
For more details, you could check [here](https://github.com/FunAudioLLM/CosyVoice/tree/main/runtime/triton_trtllm)

---

## 🚀 生产部署 (Production Deployment)

本仓库提供了**两套生产级 Docker 部署方案**，针对不同硬件：

| 方案 | 硬件 | 推理引擎 | 主入口 | 实测 RTF (Chinese, 14-42字) |
| :--- | :--- | :--- | :--- | :--- |
| **GPU 版** | NVIDIA L20 / A800 | PyTorch + (可选)TorchScript JIT | `app.py` | 0.3-0.5 |
| **NPU 版** | 华为昇腾 910B | vLLM-Ascend (v5 默认) | `app_npu.py` | **0.56-0.72** |

两套方案都通过 FastAPI 暴露**统一的 TTS API**（`/tts`、`/tts/zero_shot`、`/tts/stream`、`/speakers/*`、`/health`），客户端代码完全通用。

---

### 📋 通用前置准备 (Pre-flight Checklist)

无论 GPU 还是 NPU 部署，以下步骤都必须先完成：

#### 1️⃣ 克隆仓库并初始化子模块

```bash
git clone <this-repo-url> CosyVoice
cd CosyVoice
git submodule update --init --recursive    # 拉取 Matcha-TTS 子模块（必须！）
```

> ⚠️ **重要**：`third_party/Matcha-TTS` 是 git submodule，缺失会导致 import 报错。

#### 2️⃣ 下载预训练模型 (推荐 modelscope 加速)

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

#### 3️⃣ 准备 wetext 缓存（文本归一化，离线运行）

`app.py` / `app_npu.py` 用 monkey-patch 让 wetext 走本地缓存，避免运行时联网下载。

```bash
# 创建宿主机缓存目录（任何位置都可以，下面以容器内默认路径为例）
mkdir -p ~/.cache/modelscope/hub/pengzhendong/wetext

# 提前下载 wetext (一次性，约 800MB)
modelscope download --model pengzhendong/wetext \
  --local_dir ~/.cache/modelscope/hub/pengzhendong/wetext
```

#### 4️⃣ 准备说话人音频 (用于 zero-shot 克隆)

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

### 🟢 GPU 部署 (NVIDIA L20 / A800 / 4090 / H100)

#### 镜像构建

```bash
# 基于 nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04，含 conda + Python 3.10
docker compose build           # 用 Dockerfile + docker-compose.yml
```

构建用时约 15-30 分钟（取决于网络）。镜像名 `cosyvoice-tts-api:latest`，约 18 GB。

#### docker-compose.yml 关键挂载

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

#### GPU 启动命令

```bash
# 第 1 张 GPU、默认端口 8080
docker compose up -d

# 自定义：用第 2 张 GPU、端口 8090、并发 8
GPU_DEVICE_ID=1 API_PORT=8090 MAX_CONCURRENCY=8 docker compose up -d

# 查看日志、等待预热完成
docker compose logs -f cosyvoice-tts
```

启动后 60-180 秒（取决于 GPU），日志看到 `预热全部完成, 服务就绪` 即可调用。

#### GPU 自检

```bash
curl http://localhost:8080/health
curl "http://localhost:8080/tts?text=你好世界&name=10003-女声" --output test.wav
```

---

### 🟣 NPU 部署 (华为昇腾 910B) ★

> 实测 RTF **0.56-0.72**（Chinese, vLLM-Ascend 0.17 + CosyVoice2-0.5B + 910B 单卡）。

#### NPU 镜像构建

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

#### docker-compose.npu.yml 关键挂载

NPU 部署有**两类**挂载：宿主机 NPU 驱动（必须，否则 NPU 无法访问）+ 项目文件。

##### A. 宿主机 NPU 驱动挂载（**必须，全部 6 条**）

```yaml
- /usr/local/dcmi:/usr/local/dcmi
- /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool
- /usr/local/bin/npu-smi:/usr/local/bin/npu-smi
- /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/
- /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info
- /etc/ascend_install.info:/etc/ascend_install.info
```

> 这 6 条让容器能调用宿主机的 NPU 驱动 + 工具。**任何一条缺失都会导致 NPU 探测失败。**

##### B. NPU 设备文件映射（**必须**）

```yaml
devices:
  - /dev/davinci${NPU_DEVICE_ID:-0}:/dev/davinci${NPU_DEVICE_ID:-0}    # 用哪张卡
  - /dev/davinci_manager:/dev/davinci_manager
  - /dev/devmm_svm:/dev/devmm_svm
  - /dev/hisi_hdc:/dev/hisi_hdc
```

##### C. 项目挂载

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

#### 核心环境变量 (NPU 专属)

| 变量 | 默认 | 说明 |
| :--- | :--- | :--- |
| `LOAD_VLLM` | **`true`** ★ | **v5 主推 vLLM-Ascend**，无早停 bug |
| `NPU_DEVICE_ID` | `0` | 用第几张 NPU 卡 (`/dev/davinci0`) |
| `ASCEND_RT_VISIBLE_DEVICES` | `0` | CANN runtime 卡可见性，**必须** = `NPU_DEVICE_ID` |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | NPU 多进程必须用 spawn（fork 会崩） |
| `VLLM_USE_V1` | `1` | 强制 vLLM V1 引擎 |
| `FP16` | `true` | NPU 910B 推荐 FP16 |
| `MAX_CONCURRENCY` | `4` | NPU 910B 单卡建议 4-8 |

##### Flow 优化（vLLM 模式下不生效，fallback 模式生效）

| 变量 | 默认 | 说明 |
| :--- | :--- | :--- |
| `NPU_FLOW_TIMESTEPS` | `4` | Flow ODE 步数（10→4 砍 60%） |
| `NPU_DISABLE_CFG` | `true` | CFG-free（batch 2→1） |

##### Fallback 路径专用（**默认全关**，仅在 `LOAD_VLLM=false` 时生效）

| 变量 | 默认 | 说明 |
| :--- | :--- | :--- |
| `NPU_FUSED_ATTN` | `false` | ⚠️ 自研 fused attention（**有早停 bug，不要开**） |
| `NPU_PROFILE` | `false` | 模块级耗时诊断 |

#### NPU 启动命令

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

#### NPU 冒烟自检 (强烈推荐)

容器跑起来后，跑一次冒烟脚本验证 vLLM 全链路：

```bash
docker exec -it cosyvoice-tts \
  python /workspace/CosyVoice/scripts/smoke_vllm_npu.py
```

会按 6 步检查：torch_npu / vllm import / vllm-ascend backend / CosyVoice2 注册 / 模型目录 / EngineArgs。全绿就放心。

#### NPU 性能基线 (v5 vLLM-Ascend, 910B 单卡)

| 文本字数 | speech_len | LLM 时间 | RTF |
| :---: | :---: | :---: | :---: |
| 14 | 3.84s | ~1s | **0.72** |
| 29 | 6.0s | ~2s | **0.63** |
| 42 | 10.72s | ~4s | **0.56** |

> 对比 v3/v4 自研 fused attention：speech_len 仅为正常的 24-50%（早停 bug）。**v5 切到 vLLM 后早停彻底消失，质量无损。**

#### NPU Fallback 路径（vLLM 跑不通时）

```bash
# 关掉 vLLM，回退到 v4 npu_optimizer（仅 Flow 优化有效）
LOAD_VLLM=false docker compose -f docker-compose.npu.yml up -d
```

⚠️ 即便回退，**也不要打开 `NPU_FUSED_ATTN=true`** — 已实证有早停 bug。

---

### 🔌 客户端调用示例

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

### 🛠️ 常见故障排查

| 现象 | 原因 | 解决 |
| :--- | :--- | :--- |
| `Matcha-TTS` import 报错 | git submodule 未拉 | `git submodule update --init --recursive` |
| wetext 启动卡在下载 | 缺 wetext 缓存挂载 | 提前下载并挂载 `${WETEXT_CACHE_DIR}` |
| GPU 容器起不来 | nvidia-container-toolkit 未装 | `apt install nvidia-container-toolkit && systemctl restart docker` |
| NPU `torch.npu.is_available()=False` | 6 条驱动挂载缺一 / `ASCEND_RT_VISIBLE_DEVICES` 错 | 跑 `scripts/smoke_vllm_npu.py` 定位 |
| NPU 推理音频太短（早停 bug） | 走了 fallback 且开了 `NPU_FUSED_ATTN` | 关掉 `NPU_FUSED_ATTN=false`，确保 `LOAD_VLLM=true` |
| vLLM 模型导出失败 | 模型目录权限/磁盘空间 | 确保 `pretrained_models/CosyVoice2-0.5B/` 可写、有 2GB 空间放 `vllm/` 子目录 |

## Discussion & Communication

You can directly discuss on [Github Issues](https://github.com/FunAudioLLM/CosyVoice/issues).

You can also scan the QR code to join our official Dingding chat group.

<img src="./asset/dingding.png" width="250px">

## Acknowledge

1. We borrowed a lot of code from [FunASR](https://github.com/modelscope/FunASR).
2. We borrowed a lot of code from [FunCodec](https://github.com/modelscope/FunCodec).
3. We borrowed a lot of code from [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS).
4. We borrowed a lot of code from [AcademiCodec](https://github.com/yangdongchao/AcademiCodec).
5. We borrowed a lot of code from [WeNet](https://github.com/wenet-e2e/wenet).

## Citations

``` bibtex
@article{du2024cosyvoice,
  title={Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens},
  author={Du, Zhihao and Chen, Qian and Zhang, Shiliang and Hu, Kai and Lu, Heng and Yang, Yexin and Hu, Hangrui and Zheng, Siqi and Gu, Yue and Ma, Ziyang and others},
  journal={arXiv preprint arXiv:2407.05407},
  year={2024}
}

@article{du2024cosyvoice,
  title={Cosyvoice 2: Scalable streaming speech synthesis with large language models},
  author={Du, Zhihao and Wang, Yuxuan and Chen, Qian and Shi, Xian and Lv, Xiang and Zhao, Tianyu and Gao, Zhifu and Yang, Yexin and Gao, Changfeng and Wang, Hui and others},
  journal={arXiv preprint arXiv:2412.10117},
  year={2024}
}

@article{du2025cosyvoice,
  title={CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training},
  author={Du, Zhihao and Gao, Changfeng and Wang, Yuxuan and Yu, Fan and Zhao, Tianyu and Wang, Hao and Lv, Xiang and Wang, Hui and Shi, Xian and An, Keyu and others},
  journal={arXiv preprint arXiv:2505.17589},
  year={2025}
}

@inproceedings{lyu2025build,
  title={Build LLM-Based Zero-Shot Streaming TTS System with Cosyvoice},
  author={Lyu, Xiang and Wang, Yuxuan and Zhao, Tianyu and Wang, Hao and Liu, Huadai and Du, Zhihao},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--2},
  year={2025},
  organization={IEEE}
}
```

## Disclaimer
The content provided above is for academic purposes only and is intended to demonstrate technical capabilities. Some examples are sourced from the internet. If any content infringes on your rights, please contact us to request its removal.
