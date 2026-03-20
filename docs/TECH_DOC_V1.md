# CosyVoice TTS 服务 v1 技术文档

> 版本: v1.0.0  
> 日期: 2026-03-19  
> 模型: CosyVoice2-0.5B  
> 框架: FastAPI + uvicorn

---

## 目录

1. [项目概述](#1-项目概述)
2. [CosyVoice 模型家族](#2-cosyvoice-模型家族)
3. [文本前端: wetext / ttsfrd](#3-文本前端-wetext--ttsfrd)
4. [项目结构](#4-项目结构)
5. [macOS 本地部署](#5-macos-本地部署)
6. [Linux GPU 服务器部署](#6-linux-gpu-服务器部署)
7. [Docker 镜像构建](#7-docker-镜像构建)
8. [M3 Pro 构建 AMD64 镜像](#8-m3-pro-构建-amd64-镜像)
9. [说话人管理](#9-说话人管理)
10. [常见问题与依赖解决](#10-常见问题与依赖解决)
11. [性能测试与优化](#11-性能测试与优化)
12. [运维手册](#12-运维手册)

---

## 1. 项目概述

基于阿里 CosyVoice2-0.5B 模型的生产级 TTS HTTP API 服务。支持 zero_shot 声音克隆、流式合成、说话人预注册与复用。

### 核心特性

- 兼容原 ChatTTS app.py 接口参数, 无缝替换
- 说话人 .wav + .txt 预注册, 启动时自动加载
- 流式合成 (is_stream=1), 首包延迟 ~150ms (GPU)
- 并发控制 (asyncio.Semaphore), 防止 GPU OOM
- GPU 优化: JIT / TensorRT / vLLM / FP16

### 技术栈

| 组件 | 技术 |
|------|------|
| 推理引擎 | CosyVoice2-0.5B (zero_shot) |
| API 框架 | FastAPI + uvicorn |
| 音频处理 | torchaudio + soundfile |
| 容器化 | Docker + uv (替代 conda) |
| GPU 加速 | CUDA 12.4 + cuDNN |

---

## 2. CosyVoice 模型家族

CosyVoice 是阿里 FunAudioLLM 团队开源的多语言语音合成大模型，经历了三代演进。核心架构为"LLM + Flow Matching"两阶段：文本先经 LLM 生成语音 token，再由 Flow Matching 解码为音频波形。

### 2.1 版本演进

| 版本 | 发布时间 | 参数量 | 核心改进 |
|------|---------|--------|---------|
| CosyVoice 1.0 | 2024.07 | 300M | 奠基版本，监督式语义 token + Flow Matching |
| CosyVoice 2.0 | 2024.12 | 0.5B | 流式合成、有限标量量化 (FSQ)、预训练 LLM 骨干 |
| Fun-CosyVoice 3.0 | 2025.12 | 0.5B/1.5B | 强化学习后训练 (DiffRO)、发音修正、9 语言 + 18 种方言 |

### 2.2 可下载模型列表

| 模型 | 参数量 | 采样率 | 推理模式 | 适用场景 |
|------|--------|--------|---------|---------|
| **CosyVoice-300M** | 300M | 22050Hz | zero_shot, cross_lingual | 轻量部署、边缘设备 (4GB VRAM) |
| **CosyVoice-300M-SFT** | 300M | 22050Hz | sft | 固定音色快速合成 (中文女/男/英文女/男/日语男/粤语女/韩语女) |
| **CosyVoice-300M-Instruct** | 300M | 22050Hz | sft, instruct | 自然语言控制语气/情感/方言 |
| **CosyVoice2-0.5B** | 0.5B | 24000Hz | zero_shot, instruct2, cross_lingual | **生产推荐**，流式合成、声音克隆 |
| **Fun-CosyVoice3-0.5B** | 0.5B | 24000Hz | zero_shot, instruct, cross_lingual | 最高质量，9 语言 + 发音修正 |

### 2.3 各模型详细介绍

#### CosyVoice-300M (基础版)

CosyVoice 1.0 的基础模型，支持 zero_shot 声音克隆和跨语言合成。使用 14 层 TransformerLM 骨干，25Hz 帧率。需要上传参考音频 + 参考文本才能合成。不支持 SFT 预设音色，不支持 instruct 指令控制。适合资源受限环境，4GB 显存即可运行。

支持语言：中文、英文、日语、粤语、韩语。

```python
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')
# zero_shot
for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech_16k)):
    torchaudio.save(f'output_{i}.wav', j['tts_speech'], 22050)
```

#### CosyVoice-300M-SFT (预设音色版)

在 300M 基础上微调，内置 7 个预设说话人音色，无需参考音频即可使用。不支持 zero_shot 和 instruct。适合只需要固定音色的场景。

内置音色：中文女、中文男、英文女、英文男、日语男、粤语女、韩语女。

```python
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT')
print(cosyvoice.list_available_spks())  # ['中文女', '中文男', ...]
for i, j in enumerate(cosyvoice.inference_sft('你好世界', '中文女')):
    torchaudio.save(f'sft_{i}.wav', j['tts_speech'], 22050)
```

#### CosyVoice-300M-Instruct (指令控制版)

在 SFT 基础上增加 instruct 能力，可通过自然语言指令控制合成风格：语速、情感、方言等。同时保留 SFT 预设音色。

```python
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-Instruct')
# instruct: 用自然语言控制风格
for i, j in enumerate(cosyvoice.inference_instruct(
    '在面对挑战时，他展现了非凡的勇气与智慧。',
    '中文男',
    'Theo is a fiery, passionate rebel leader.',
)):
    torchaudio.save(f'instruct_{i}.wav', j['tts_speech'], 22050)
```

#### CosyVoice2-0.5B (第二代，本项目默认)

CosyVoice 2.0 的核心模型，参数量提升到 0.5B，采样率 24000Hz。关键改进包括：有限标量量化 (FSQ) 提升 codebook 利用率、预训练 LLM 作为骨干网络、chunk-aware 因果 Flow Matching 同时支持流式和非流式合成。

不支持 SFT 预设音色（没有内置说话人），所有推理走 zero_shot 模式。支持 instruct2 指令控制。支持 vLLM 加速（4x）和 TensorRT 加速（2-3x）。

这是本项目 `app.py` 的默认模型。

```python
cosyvoice = AutoModel('pretrained_models/CosyVoice2-0.5B')
# zero_shot with registered speaker
cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech, 'my_spk')
for i, j in enumerate(cosyvoice.inference_zero_shot(
    text, '', '', zero_shot_spk_id='my_spk'
)):
    torchaudio.save(f'output_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

#### Fun-CosyVoice3-0.5B (第三代，最高质量)

CosyVoice 3.0 版本，在 2.0 基础上引入强化学习后训练 (DiffRO)、Diffusion Transformer (DiT) 解码器、数据规模扩大到百万小时。

核心提升：内容一致性 CER 从 1.45% 降至 0.81%、说话人相似度从 74.8% 提升至 77.4%。新增发音修正 (Pronunciation Inpainting) 功能，支持拼音/音素级别干预。语言覆盖扩展到 9 种语言 + 18 种中国方言。

```python
cosyvoice = AutoModel('pretrained_models/Fun-CosyVoice3-0.5B')
# zero_shot
for i, j in enumerate(cosyvoice.inference_zero_shot(
    '八百标兵奔北坡，北坡炮兵并排跑。',
    'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
    './asset/zero_shot_prompt.wav',
)):
    torchaudio.save(f'output_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)

# 发音修正 (CosyVoice3 独有)
for i, j in enumerate(cosyvoice.inference_zero_shot(
    '高管也通过电话、短信、微信等方式对报道[j][ǐ]予好评。',  # 拼音标注修正 "给" 的发音
    prompt_text, prompt_speech,
)):
    torchaudio.save(f'hotfix_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

### 2.4 推理模式能力矩阵

| 推理模式 | 说明 | 300M | 300M-SFT | 300M-Instruct | V2-0.5B | V3-0.5B |
|---------|------|------|----------|---------------|---------|---------|
| `inference_sft` | 预设音色合成 | ❌ | ✅ | ✅ | ❌ | ❌ |
| `inference_zero_shot` | 声音克隆 | ✅ | ❌ | ❌ | ✅ | ✅ |
| `inference_instruct` | 自然语言指令 (v1) | ❌ | ❌ | ✅ | ❌ | ✅ |
| `inference_instruct2` | 自然语言指令 (v2) | ❌ | ❌ | ❌ | ✅ | ❌ |
| `inference_cross_lingual` | 跨语言合成 | ✅ | ❌ | ❌ | ✅ | ✅ |
| 流式合成 (`stream=True`) | 低延迟 | ✅ | ✅ | ✅ | ✅ | ✅ |

### 2.5 模型选择建议

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| 生产环境声音克隆 | **CosyVoice2-0.5B** | 稳定、流式支持好、vLLM/TRT 加速成熟 |
| 最高合成质量 | **Fun-CosyVoice3-0.5B** | CER 最低、发音修正、多语言最佳 |
| 实时流式场景 | **CosyVoice2-0.5B + vLLM** | 首包延迟 ~150ms |
| 资源受限/边缘设备 | **CosyVoice-300M** | 4GB 显存、300M 参数 |
| 固定音色快速合成 | **CosyVoice-300M-SFT** | 无需参考音频 |
| 情感/风格控制 | **CosyVoice-300M-Instruct** | 自然语言指令 |

### 2.6 模型下载

```python
from modelscope import snapshot_download

# CosyVoice2-0.5B (本项目默认)
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')

# Fun-CosyVoice3-0.5B (最高质量)
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')

# CosyVoice-300M 系列
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='pretrained_models/CosyVoice-300M-Instruct')

# 文本前端资源 (可选, 提升数字/符号归一化精度)
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```

---

## 3. 文本前端: wetext / ttsfrd

### 11.1 作用

CosyVoice 的推理流程中，文本需要先经过"文本前端"处理再送入 LLM。文本前端负责：

- **文本归一化 (Text Normalization, TN)**: 将数字、符号、缩写转为可读文字
  - `"2.5平方电线"` → `"二点五平方电线"`
  - `"666"` → `"六六六"`
  - `"2024年8月8日"` → `"二零二四年八月八日"`
- **逆文本归一化 (Inverse Text Normalization, ITN)**: TN 的反向操作
- **多音字处理**: 根据上下文选择正确读音
- **儿化音处理**: `"船新版本儿"` → `"船新版本"`

### 11.2 两种文本前端

CosyVoice 支持两种文本前端，按优先级自动选择：

| 前端 | 包名 | 依赖 | 精度 | 安装方式 |
|------|------|------|------|---------|
| **ttsfrd** | `ttsfrd` | 需要手动安装 wheel | 高 (阿里内部方案) | 见下方 |
| **wetext** | `wetext` | 自动安装 | 中 (开源方案) | `pip install wetext` |

启动日志中可看到使用哪个前端：
```
INFO: use wetext frontend       ← 使用 wetext
INFO: use ttsfrd frontend       ← 使用 ttsfrd
```

### 11.3 wetext 详解

wetext 是 WeTextProcessing 的轻量运行时，由 WeNet 团队开源。基于有限状态转换器 (FST) 实现，不依赖 Pynini（Pynini 是 WeTextProcessing 原始版的依赖，安装困难）。

支持语言：中文、英文、日语。

```python
from wetext import Normalizer

# 中文 TN
normalizer = Normalizer(lang="zh", operator="tn")
normalizer.normalize("你好 WeTextProcessing 1.0，简直666")
# → "你好 WeTextProcessing 一点零，简直六六六"

# 中文 TN (去儿化音)
normalizer = Normalizer(lang="zh", operator="tn", remove_erhua=True)
normalizer.normalize("全新版本儿")
# → "全新版本"

# 英文 TN
normalizer = Normalizer(lang="en", operator="tn")
normalizer.normalize("Today is August 8, 2024.")
# → "Today is the eighth of august, twenty twenty four."

# 中文 ITN
normalizer = Normalizer(lang="zh", operator="itn")
normalizer.normalize("二点五平方电线")
# → "2.5平方电线"
```

wetext 的 Normalizer 参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lang` | str | `"auto"` | 语言: `"auto"`, `"zh"`, `"en"`, `"ja"` |
| `operator` | str | `"tn"` | 操作: `"tn"` (归一化) / `"itn"` (逆归一化) |
| `remove_erhua` | bool | `False` | 去除儿化音 |
| `traditional_to_simple` | bool | `False` | 繁体转简体 |
| `full_to_half` | bool | `False` | 全角转半角 |
| `remove_puncts` | bool | `False` | 去除标点 |
| `tag_oov` | bool | `False` | 标记未登录词 |

### 11.4 ttsfrd (阿里内部方案)

ttsfrd 是阿里通义实验室的文本前端方案，精度高于 wetext，但需要手动安装 wheel 包：

```bash
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```

限制：仅支持 Linux x86_64 + Python 3.10。macOS 和其他 Python 版本无法使用。

### 11.5 在本项目中的处理

CosyVoice 模型启动时自动检测文本前端：ttsfrd 可用则优先使用，否则 fallback 到 wetext。本项目不需要额外配置。

wetext 的模型数据首次启动时会自动下载到 `~/.cache/modelscope/hub/pengzhendong/wetext`。Docker 部署时通过 volume 挂载避免重复下载：

```yaml
volumes:
  - ${WETEXT_CACHE_DIR}:/root/.cache/modelscope/hub/pengzhendong/wetext
```

---

## 4. 项目结构

```
CosyVoice/
├── app.py                  # API 服务主文件 (CosyVoice2 专用)
├── app_all_model.py        # 全模型兼容版 (自动检测模型能力)
├── test_tts.py             # 性能测试脚本
│
├── Dockerfile              # Docker 镜像 (conda 版)
├── Dockerfile.uv           # Docker 镜像 (uv 版, 推荐)
├── Dockerfile.gpu-full     # 叠加 TensorRT + vLLM
├── docker-compose.yml      # conda 版编排
├── docker-compose.uv.yml   # uv 版编排
├── .dockerignore           # 排除 .venv, pretrained_models 等
├── .env.example            # 环境变量模板
│
├── speakers/               # 说话人文件目录
│   ├── 1111-男声.wav       #   参考音频
│   ├── 1111-男声.txt       #   音频对应文字
│   ├── 10003-女声.wav
│   └── 10003-女声.txt
│
├── static/wavs/            # 生成音频输出 (通过 HTTP 访问)
├── uploads/                # 临时上传目录
├── logs/                   # 日志
│
├── pretrained_models/      # 模型 (不打进 Docker 镜像)
│   └── CosyVoice2-0.5B/
│
├── cosyvoice/              # CosyVoice 模型源码
├── third_party/            # 第三方依赖 (Matcha-TTS)
└── requirements.txt        # Python 依赖
```

---

## 5. macOS 本地部署

### 11.1 环境要求

- macOS 13+ (Apple Silicon M1/M2/M3)
- Python 3.11
- ffmpeg
- 16GB+ 内存

### 11.2 安装步骤

```bash
# 1. 克隆项目
git clone --recursive https://github.com/zyjohn-design/CosyVoice.git
cd CosyVoice

# 2. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 安装 Python 3.11
uv python install 3.11

# 4. 创建虚拟环境
uv venv --python 3.11
source .venv/bin/activate

# 5. 安装 ffmpeg
brew install ffmpeg

# 6. 安装依赖
uv pip install -r requirements.txt --index-strategy unsafe-best-match

# 7. 下载模型
python -c "
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
"
```

### 11.3 准备说话人

```bash
mkdir -p speakers

# 放置参考音频 (3~10秒, wav, >=16kHz)
cp /path/to/reference.wav speakers/1111-男声.wav

# 创建对应文本 (音频里说的话, 必须准确)
echo "希望你以后能够做的比我还好呦。" > speakers/1111-男声.txt
```

### 11.4 启动服务

```bash
# 方式1: 直接运行
python app.py

# 方式2: uvicorn (推荐)
python -m uvicorn app:app --host 0.0.0.0 --port 8080

# 验证
curl http://localhost:8080/health
curl http://localhost:8080/speakers
curl "http://localhost:8080/tts?text=你好世界&voice=1111-男声&wav=1" -o test.wav
```

### 11.5 macOS 已知问题

| 问题 | 解决方案 |
|------|----------|
| `ffmpeg not found` | `brew install ffmpeg` |
| `TorchCodec is required` | CosyVoice 内部依赖, app.py 已用 torchaudio 做 fallback |
| `MPS fallback` | Apple Silicon 正常现象, 部分算子回退到 CPU |
| 推理速度慢 (RTF>1) | M3 Pro 正常, GPU 服务器可达 10~40x 提速 |

---

## 6. Linux GPU 服务器部署

### 12.1 环境要求

- Ubuntu 22.04
- NVIDIA Driver 535+
- CUDA 12.4
- Docker 27+ (含 NVIDIA Container Toolkit)
- GPU: L20 (48GB) / A800 (80GB)

### 12.2 Docker 部署 (推荐)

```bash
# 1. 传代码到服务器
rsync -avP --exclude='pretrained_models/' --exclude='.git/' --exclude='.venv/' \
  ./ user@gpu-server:/opt/cosyvoice/

# 2. SSH 到服务器
ssh user@gpu-server
cd /opt/cosyvoice

# 3. 下载模型
mkdir -p pretrained_models
python3 -c "
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
"

# 4. 准备环境
cp .env.example .env
vim .env  # 改 WETEXT_CACHE_DIR 等

# 5. 构建镜像 (uv 版)
docker build -t cosyvoice-tts-api:latest -f Dockerfile.uv .

# 6. 启动
docker compose -f docker-compose.uv.yml up -d
docker compose -f docker-compose.uv.yml logs -f
```

### 12.3 裸机部署

```bash
# 1. 安装 uv + Python
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate

# 2. 安装系统依赖
sudo apt install ffmpeg sox libsox-dev

# 3. 安装 Python 依赖
uv pip install -r requirements.txt --index-strategy unsafe-best-match
uv pip install torchcodec python-multipart

# 4. 启动
MODEL_DIR=pretrained_models/CosyVoice2-0.5B \
LOAD_JIT=true FP16=true \
python -m uvicorn app:app --host 0.0.0.0 --port 8080
```

---

## 7. Docker 镜像构建

### 11.1 文件说明

| 文件 | 说明 |
|------|------|
| `Dockerfile` | conda 版, 兼容官方方案 |
| `Dockerfile.uv` | **uv 版 (推荐)**, 构建快, 镜像小 |
| `Dockerfile.gpu-full` | 在已有镜像上叠加 TensorRT + vLLM |
| `.dockerignore` | 排除 .venv, pretrained_models, .git 等 |

### 11.2 构建命令

```bash
# uv 版 (推荐)
docker build -t cosyvoice-tts-api:latest -f Dockerfile.uv .

# 叠加 TensorRT + vLLM
docker build -t cosyvoice-tts-api:gpu-full -f Dockerfile.gpu-full .

# 只叠加 TensorRT
docker build -t cosyvoice-tts-api:trt -f Dockerfile.gpu-full --target trt .
```

### 11.3 缓存机制

Dockerfile.uv 使用 `--mount=type=cache,target=/root/.cache/uv` 缓存下载的包:

| 场景 | 耗时 |
|------|------|
| 首次构建 | ~20 分钟 |
| 修改代码后重新构建 | **~30 秒** (包全部命中缓存) |
| 修改 requirements.txt | ~5 分钟 (只下载新增的包) |

### 11.4 .dockerignore

```
.venv/
pretrained_models/
.git/
__pycache__/
static/wavs/
uploads/
logs/
*.tar
*.tar.gz
.DS_Store
```

> 模型通过 docker-compose volumes 挂载, 不打进镜像。

---

## 8. M3 Pro 构建 AMD64 镜像

### 12.1 方案对比

| 方案 | 耗时 | 推荐 |
|------|------|------|
| M3 上 buildx 跨平台构建 | 1~3 小时 (QEMU 模拟) | ❌ |
| **服务器上直接构建** | **10~20 分钟** | **✅** |
| 仓库中转 | 取决于网速 | CI/CD 场景 |

### 12.2 推荐: 服务器直接构建

```bash
# Mac 上传代码
rsync -avP --exclude='pretrained_models/' --exclude='.git/' --exclude='.venv/' \
  ./ user@gpu-server:/opt/cosyvoice/

# 服务器上构建
ssh user@gpu-server
cd /opt/cosyvoice
docker build -t cosyvoice-tts-api:latest -f Dockerfile.uv .
```

### 12.3 M3 跨平台构建 (备用)

```bash
# 创建 buildx 构建器
docker buildx create --name cosyvoice-builder --platform linux/amd64 --use
docker buildx inspect --bootstrap

# 构建并导出 tar
docker buildx build \
  --platform linux/amd64 \
  --tag cosyvoice-tts-api:latest \
  --output type=docker,dest=cosyvoice-tts-api-amd64.tar \
  -f Dockerfile.uv .

# 传到服务器并加载
scp cosyvoice-tts-api-amd64.tar user@gpu-server:/opt/
ssh user@gpu-server "docker load -i /opt/cosyvoice-tts-api-amd64.tar"
```

### 12.4 导出/导入已有镜像

```bash
# 导出
docker save cosyvoice-tts-api:latest -o cosyvoice-tts-api.tar
gzip cosyvoice-tts-api.tar

# 传输
rsync -avP cosyvoice-tts-api.tar.gz user@gpu-server:/opt/

# 导入
ssh user@gpu-server
gunzip cosyvoice-tts-api.tar.gz
docker load -i cosyvoice-tts-api.tar
```

---

## 9. 说话人管理

### 11.1 目录结构

```
speakers/
├── 1111-男声.wav     ← 参考音频 (3~10秒, wav, >=16kHz)
├── 1111-男声.txt     ← 音频中说的话 (必须准确!)
├── 10003-女声.wav
└── 10003-女声.txt
```

### 11.2 关键要求

- **prompt_text 必须准确**: `.txt` 中的文字必须与 `.wav` 音频内容完全一致。不一致会导致模型文本-音频对齐错误, 合成的语音内容乱码。
- **音频质量**: 清晰语音, 无背景噪音, 3~10 秒, 采样率 >=16kHz。
- **文件名对应**: `张三.wav` 和 `张三.txt` 必须同名。

### 11.3 注册流程

```
服务启动
  → 扫描 speakers/ 下所有 .wav 文件
  → 读取同名 .txt 作为 prompt_text
  → 调用 cosyvoice_model.add_zero_shot_spk(prompt_text, audio, spk_id)
  → 注册到 _registered_spk_ids 集合
  → 推理时通过 zero_shot_spk_id=spk_id 复用
```

### 11.4 运行时管理

```bash
# 列出
curl http://localhost:8080/speakers

# API 上传注册
curl -X POST http://localhost:8080/speakers/upload \
  -F "name=新说话人" \
  -F "audio=@ref.wav" \
  -F "prompt_text=参考音频中的文字"

# 删除
curl -X DELETE http://localhost:8080/speakers/新说话人

# 通过 /tts/zero_shot 克隆并保存
curl -X POST http://localhost:8080/tts/zero_shot \
  -F "text=测试文本" \
  -F "prompt_text=参考文字" \
  -F "prompt_audio=@ref.wav" \
  -F "save_speaker=新说话人" \
  -o output.wav
```

---

## 10. 常见问题与依赖解决

### 12.1 ffmpeg not found

```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg

# Docker 镜像中已包含
```

### 12.2 TorchCodec is required

CosyVoice 新版本的 `load_wav` 和 `add_zero_shot_spk` 内部使用 torchcodec。

```bash
# 安装
pip install torchcodec

# macOS ARM 可能装不上, app.py 已用 torchaudio 做 fallback
# Docker 镜像中已包含
```

### 12.3 openai-whisper 构建失败 (ModuleNotFoundError: pkg_resources)

新版 setuptools (>=71) 将 pkg_resources 拆出。

```bash
# 解决: 先装旧版 setuptools
pip install "setuptools<71"
pip install openai-whisper==20231117
```

Dockerfile.uv 中已处理: `--no-build-isolation-package openai-whisper`

### 12.4 tensorrt-cu12-libs 构建极慢/卡住

该包 ~900MB, 源码构建耗时长。

```bash
# 方案1: 跳过 (部署后按需装)
grep -v "^tensorrt" requirements.txt > req_no_trt.txt
pip install -r req_no_trt.txt

# 方案2: 在 GPU 服务器上装 (直接下载 wheel, 不需要编译)
pip install tensorrt-cu12==10.13.3.9 tensorrt-cu12-bindings==10.13.3.9 tensorrt-cu12-libs==10.13.3.9
```

### 12.5 pynini 安装失败

pynini 是 WeTextProcessing 的依赖, conda 通常能装上, uv/pip 可能失败。

```bash
# conda
conda install -c conda-forge pynini==2.1.5

# 装不上也没关系, 会 fallback 到 wetext
```

### 12.6 Docker 拉取镜像证书错误

```
tls: failed to verify certificate: x509: certificate is valid for *.atlassolutions.com
```

网络代理/VPN 拦截了 Docker Hub 的 TLS 连接。

```bash
# 方案1: Docker Desktop → Settings → Docker Engine → 添加镜像源
{
  "registry-mirrors": ["https://docker.1ms.run"]
}

# 方案2: 手动拉取基础镜像
docker pull nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
```

### 12.7 voice 不匹配 / 合成内容乱码

根因: `speakers/xxx.txt` 中的 prompt_text 与 `.wav` 音频内容不一致。

解决: 确保 `.txt` 中的文字是音频里**说的话的准确文字**。

---

## 11. 性能测试与优化

### 11.1 测试脚本

```bash
pip install requests
python test_tts.py --host 127.0.0.1 --port 8080
```

### 11.2 M3 Pro 基准测试 (CosyVoice2-0.5B)

| 字数 | 推理(s) | 音频(s) | RTF | 字/秒 |
|------|--------|--------|-----|-------|
| 5 | 2.38 | 1.08 | 2.204 | 2.1 |
| 10 | 4.61 | 3.04 | 1.516 | 2.6 |
| 20 | 7.95 | 6.40 | 1.242 | 3.4 |
| 30 | 9.95 | 8.24 | 1.208 | 3.9 |
| 50 | 13.95 | 11.52 | 1.211 | 4.4 |

> RTF = 推理时间 ÷ 音频时长。RTF < 1 = 比实时快。

### 11.3 GPU 性能预测

| 配置 | 50字预计 | RTF | 提升倍数 |
|------|---------|-----|---------|
| M3 Pro (当前) | ~14s | 1.26 | 1x |
| L20 fp16 | ~1.5s | 0.12 | **10x** |
| L20 fp16+JIT+TRT | ~0.8s | 0.06 | **18x** |
| A800 fp16+JIT+TRT+vLLM | ~0.4s | 0.03 | **35x** |

### 11.4 GPU 优化参数

通过环境变量控制:

| 环境变量 | 作用 | 提升 |
|---------|------|------|
| `FP16=true` | 半精度推理 | +50% 速度, -50% 显存 |
| `LOAD_JIT=true` | TorchScript | +10~15% |
| `LOAD_TRT=true` | TensorRT 加速 Flow Matching | +2~3x |
| `LOAD_VLLM=true` | vLLM 加速 LLM decoding | +4x |

```bash
# .env 配置
FP16=true
LOAD_JIT=true
LOAD_TRT=false    # 需先构建 .plan 文件
LOAD_VLLM=false   # 需额外安装 vllm==v0.9.0
MAX_CONCURRENCY=4  # L20: 4-6, A800: 8-12
```

### 11.5 vLLM 安装

```bash
# 在容器内
pip install vllm==v0.9.0 transformers==4.51.3 numpy==1.26.4

# 或用 Dockerfile.gpu-full 构建
docker build -t cosyvoice-tts-api:gpu-full -f Dockerfile.gpu-full .
```

### 11.6 TensorRT 引擎构建

```bash
trtexec \
  --onnx=pretrained_models/CosyVoice2-0.5B/flow.decoder.estimator.fp32.onnx \
  --saveEngine=pretrained_models/CosyVoice2-0.5B/flow.decoder.estimator.fp32.plan \
  --minShapes=x:1x80x1,mask:1x1x1,mu:1x80x1,t:1,spks:1x80,cond:1x80x1 \
  --maxShapes=x:1x80x4096,mask:1x1x4096,mu:1x80x4096,t:1,spks:1x80,cond:1x80x4096
```

---

## 12. 运维手册

### 12.1 服务管理

```bash
# 启动
docker compose -f docker-compose.uv.yml up -d

# 查看日志
docker compose -f docker-compose.uv.yml logs -f

# 重启
docker compose -f docker-compose.uv.yml restart

# 停止
docker compose -f docker-compose.uv.yml down

# 查看状态
docker compose -f docker-compose.uv.yml ps
```

### 12.2 健康检查

```bash
# API 检查
curl http://localhost:8080/health

# Docker 检查
docker inspect --format='{{.State.Health.Status}}' cosyvoice-tts
```

### 12.3 磁盘清理

```bash
# 清理生成的音频
curl -X POST http://localhost:8080/clear_wavs

# 清理 Docker
docker system prune -f
```

### 12.4 更新部署

```bash
# 1. 更新代码
rsync -avP --exclude='pretrained_models/' ./ user@server:/opt/cosyvoice/

# 2. 重新构建
docker build -t cosyvoice-tts-api:latest -f Dockerfile.uv .

# 3. 重启
docker compose -f docker-compose.uv.yml up -d
```

### 12.5 GPU 监控

```bash
# 实时 GPU 使用
watch -n 1 nvidia-smi

# 容器内 GPU 使用
docker exec cosyvoice-tts nvidia-smi
```
