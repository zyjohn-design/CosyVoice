# CLAUDE.md

## Project Overview

CosyVoice is a multilingual zero-shot text-to-speech (TTS) system based on large language models (LLM), developed by Alibaba FunAudioLLM. It supports three model generations: CosyVoice 1.0 (300M), CosyVoice 2.0 (0.5B), and Fun-CosyVoice 3.0 (0.5B). The project includes both the upstream open-source TTS engine and a custom production-grade FastAPI service (`app.py`) for deployment.

## Tech Stack

- **Language**: Python 3.10+ (conda env) / Python 3.11 (uv/local venv)
- **ML Framework**: PyTorch 2.3+, torchaudio, ONNX Runtime
- **Web Framework**: FastAPI + uvicorn (production API in `app.py`), Gradio (demo in `webui.py`)
- **Package Management**: pip + requirements.txt (primary), uv + pyproject.toml (local dev alternative)
- **Deployment**: Docker + docker-compose, supports NVIDIA GPU (CUDA 12.4) and Huawei NPU (CANN)
- **Key Dependencies**: HyperPyYAML, modelscope, transformers, wetext, Matcha-TTS (git submodule)

## Project Structure

```
CosyVoice/
├── app.py                  # Production FastAPI TTS API server (main entry)
├── app_npu.py              # NPU-adapted API server (Huawei Ascend 910B)
├── app_all_model.py        # Multi-model API variant
├── example.py              # Usage examples for all 3 model generations
├── vllm_example.py         # vLLM accelerated inference example
├── webui.py                # Gradio web demo
├── test_tts.py             # API performance test script
├── cosyvoice/              # Core TTS engine package
│   ├── cli/                # High-level API
│   │   ├── cosyvoice.py    # CosyVoice/CosyVoice2/CosyVoice3/AutoModel classes
│   │   ├── model.py        # Model implementations (CosyVoiceModel, CosyVoice2Model, CosyVoice3Model)
│   │   ├── model_npu.py    # NPU-adapted model implementations
│   │   └── frontend.py     # Text frontend (tokenization, normalization)
│   ├── llm/                # LLM module (autoregressive speech token generation)
│   ├── flow/               # Flow matching module (acoustic feature generation)
│   ├── hifigan/            # HiFi-GAN vocoder (waveform synthesis)
│   ├── transformer/        # Transformer components (encoder, decoder, attention)
│   ├── tokenizer/          # Speech tokenizer
│   ├── vllm/               # vLLM integration for accelerated LLM inference
│   ├── dataset/            # Data loading and processing
│   └── utils/              # Utilities (training, masking, common functions)
├── third_party/Matcha-TTS/ # Git submodule dependency
├── pretrained_models/      # Downloaded model weights (not in git)
├── speakers/               # Speaker reference audio + text files
├── runtime/
│   ├── python/             # gRPC and FastAPI deployment runtimes
│   └── triton_trtllm/      # TensorRT-LLM + Triton deployment
├── examples/               # Training scripts (libritts, grpo, magicdata-read)
├── tools/                  # Data preprocessing tools
├── Dockerfile              # GPU production image (CUDA 12.4)
├── Dockerfile.npu          # NPU production image (Ascend)
├── Dockerfile.uv           # uv-based build image
├── docker-compose.yml      # GPU deployment compose
├── docker-compose.npu.yml  # NPU deployment compose
└── requirements.txt        # Python dependencies
```

## Architecture

The TTS pipeline has 4 stages:
1. **Frontend** (`frontend.py`): Text normalization → tokenization (via wetext or ttsfrd)
2. **LLM** (`llm/`): Autoregressive generation of speech tokens from text tokens
3. **Flow Matching** (`flow/`): Converts speech tokens to mel-spectrogram features
4. **Vocoder** (`hifigan/`): HiFi-GAN converts mel features to audio waveform

Model selection is automatic via `AutoModel()` which detects `cosyvoice.yaml` / `cosyvoice2.yaml` / `cosyvoice3.yaml` in the model directory.

## Key Classes

- `AutoModel(model_dir=...)` — Factory function, returns the correct CosyVoice/2/3 instance
- `CosyVoice` — V1 model: supports `inference_sft`, `inference_zero_shot`, `inference_cross_lingual`, `inference_instruct`, `inference_vc`
- `CosyVoice2(CosyVoice)` — V2 model: adds `inference_instruct2`, bi-streaming input
- `CosyVoice3(CosyVoice2)` — V3 model: best quality, 9 languages, pronunciation inpainting

## Inference Modes

- **SFT** (V1 only): Predefined speaker voices via `spk2info.pt`
- **Zero-shot**: Clone any voice from a reference audio + transcript
- **Cross-lingual**: Clone voice across languages
- **Instruct** (V1): Natural language control of speech style
- **Instruct2** (V2/V3): Instruction-guided synthesis with reference audio
- **Voice Conversion** (V1): Convert source audio to target speaker voice

## Development Commands

```bash
# Setup environment
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
pip install -r requirements.txt

# Run examples
python example.py

# Start Gradio web demo
python webui.py --port 50000 --model_dir pretrained_models/CosyVoice-300M

# Start production API server (local)
python app.py

# Run performance tests
python test_tts.py --host 127.0.0.1 --port 8080

# Docker deployment (GPU)
docker compose build
docker compose up -d

# Docker deployment (NPU)
docker compose -f docker-compose.npu.yml up -d
```

## Environment Variables (app.py)

| Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | `pretrained_models/CosyVoice2-0.5B` | Model directory path |
| `HOST` | `0.0.0.0` | API bind address |
| `PORT` | `8080` | API port |
| `MAX_CONCURRENCY` | `4` | Max concurrent requests |
| `LOAD_JIT` | `false` | Enable TorchScript acceleration (V1/V2 only) |
| `LOAD_TRT` | `false` | Enable TensorRT acceleration |
| `LOAD_VLLM` | `false` | Enable vLLM LLM acceleration |
| `FP16` | `false` | Enable half-precision inference |

## API Endpoints (app.py)

- `GET/POST /tts` — Main TTS endpoint (text + speaker name)
- `POST /tts/zero_shot` — Upload reference audio for voice cloning
- `POST /tts/stream` — Streaming low-latency synthesis
- `GET /speakers` — List registered speakers
- `POST /speakers/upload` — Register new speaker (upload wav + text)
- `DELETE /speakers/{name}` — Remove speaker
- `GET /health` — Health check with GPU stats

## Speaker Management

Place files in `speakers/` directory:
- `{name}.wav` — Reference audio (3-10s, ≥16kHz)
- `{name}.txt` — Transcript of the reference audio

Speakers are auto-registered on server startup via `add_zero_shot_spk()`.

## Important Notes

- `third_party/Matcha-TTS` is a git submodule; run `git submodule update --init --recursive` after cloning
- `PYTHONPATH` must include both project root and `third_party/Matcha-TTS`
- CosyVoice3 `prompt_text` requires `<|endofprompt|>` marker (auto-appended in app.py)
- V3 Japanese input must be in katakana
- Model weights are in `.gitignore`; download via modelscope/huggingface SDK
- The `wetext` package is used for text normalization when `ttsfrd` is not installed
