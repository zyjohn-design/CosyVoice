# CosyVoice TTS 华为 NPU 910B 适配方案

> 目标: 同一套代码同时支持 NVIDIA GPU 和华为 NPU 910B  
> 原则: 不修改 NVIDIA 的任何已有逻辑, NPU 适配通过运行时检测自动切换

---

## 1. 适配原理

华为昇腾 NPU 通过 `torch_npu` 库适配 PyTorch。核心机制：

```
PyTorch → torch.cuda.*  → NVIDIA GPU (CUDA)
PyTorch → torch_npu     → 华为 NPU (CANN)
```

`torch_npu` 提供两种适配方式：

| 方式 | 说明 | 侵入性 |
|------|------|--------|
| `transfer_to_npu` | 自动将 `.cuda()` / `.to("cuda")` 映射为 `.npu()` / `.to("npu")` | **零改动** |
| 手动替换 | 代码中逐个替换 `cuda` → `npu` | 高 |

**推荐用 `transfer_to_npu`**，零侵入，不影响 NVIDIA 代码。

---

## 2. 环境要求

### NPU 服务器

| 组件 | 版本 |
|------|------|
| 芯片 | Ascend 910B (32GB/64GB) |
| OS | EulerOS 2.0 / Ubuntu 20.04+ (aarch64) |
| 固件+驱动 | Ascend HDK 24.1+ |
| CANN Toolkit | 8.0+ |
| CANN Kernels | 910b 对应版本 |
| Python | 3.10 |
| PyTorch | 2.1~2.3 |
| torch_npu | 对应 PyTorch 版本 |

### 安装 CANN + torch_npu

```bash
# 1. 安装 CANN Toolkit (在 NPU 服务器上)
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/.../Ascend-cann-toolkit_8.0.RC1_linux-aarch64.run
bash Ascend-cann-toolkit_8.0.RC1_linux-aarch64.run --install

# 2. 安装 CANN Kernels
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/.../Ascend-cann-kernels-910b_8.0.RC1_linux.run
bash Ascend-cann-kernels-910b_8.0.RC1_linux.run --install

# 3. 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 4. 安装 torch_npu
pip install torch-npu==2.3.1  # 版本必须与 torch 版本对应
```

---

## 3. app.py 适配改动

只需在 app.py 的 **PyTorch 全局优化配置** 之后、**FastAPI 导入之前** 加一段设备检测代码。

### 3.1 改动位置

在现有代码中找到这段：

```python
# ── PyTorch 全局优化配置 ──
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 64
torch.set_float32_matmul_precision("high")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
```

在其后面添加：

```python
# ══════════════════════════════════════════════════════════════
#  [NPU适配] 设备自动检测: NVIDIA GPU / 华为 NPU / CPU
#  通过 torch_npu 的 transfer_to_npu 实现零侵入适配
#  所有 .cuda() / .to("cuda") 调用自动映射为 .npu() / .to("npu")
#  不影响 NVIDIA GPU 环境的任何代码
# ══════════════════════════════════════════════════════════════

DEVICE_TYPE = "cpu"  # 默认 CPU

if torch.cuda.is_available():
    # ── NVIDIA GPU ──
    DEVICE_TYPE = "cuda"
    logger.info(f"检测到 NVIDIA GPU: {torch.cuda.get_device_name(0)}")

else:
    # ── 尝试华为 NPU ──
    try:
        import torch_npu
        from torch_npu.contrib import transfer_to_npu  # 自动 cuda→npu 映射
        if torch.npu.is_available():
            DEVICE_TYPE = "npu"
            logger.info(f"检测到华为 NPU: {torch.npu.get_device_name(0)}")
            # NPU 优化配置
            # 910B 对 bfloat16 支持不完善, 强制使用 float16
            os.environ.setdefault("FP16", "true")
        else:
            logger.warning("torch_npu 已安装但未检测到 NPU 设备")
    except ImportError:
        logger.info("未检测到 GPU/NPU, 使用 CPU 推理")
```

### 3.2 /health 接口适配

现有代码只检测 CUDA，需要加 NPU 分支：

```python
@app.get("/health")
async def health():
    gpu = {}
    if DEVICE_TYPE == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "device_type": "NVIDIA GPU",
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total_mb": round(props.total_memory / 1024**2),
            "gpu_memory_used_mb": round(torch.cuda.memory_allocated(0) / 1024**2),
        }
    elif DEVICE_TYPE == "npu":
        try:
            gpu = {
                "device_type": "Huawei NPU",
                "npu_name": torch.npu.get_device_name(0),
                "npu_memory_total_mb": round(torch.npu.get_device_properties(0).total_memory / 1024**2),
                "npu_memory_used_mb": round(torch.npu.memory_allocated(0) / 1024**2),
            }
        except Exception:
            gpu = {"device_type": "Huawei NPU", "info": "available"}
    # ... 其余不变
```

### 3.3 GPU 显存清理适配

搜索所有 `torch.cuda.empty_cache()` 改为：

```python
# 原来
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 改为
if DEVICE_TYPE == "cuda":
    torch.cuda.empty_cache()
elif DEVICE_TYPE == "npu":
    torch.npu.empty_cache()
```

---

## 4. 不需要改的部分

| 模块 | 原因 |
|------|------|
| CosyVoice 模型加载 | `transfer_to_npu` 自动拦截所有 `.cuda()` 调用 |
| 推理代码 | tensor 的 `.to(device)` 被自动映射 |
| torchaudio | 不走 GPU, 在 CPU 上处理 |
| FastAPI / uvicorn | 纯 CPU 逻辑, 无关设备 |
| 说话人注册 | `add_zero_shot_spk` 内部设备自动映射 |

---

## 5. 需要关闭的功能

NPU 不支持以下 NVIDIA 专有加速：

```bash
# NPU 环境的 .env
LOAD_JIT=false        # TorchScript: NPU 部分算子不支持 JIT
LOAD_TRT=false        # TensorRT: NVIDIA 专有, NPU 不可用
LOAD_VLLM=false       # vLLM: 需要 vllm-ascend 单独适配 (见第 8 节)
FP16=true             # float16: NPU 910B 支持, 推荐开启
```

---

## 6. Docker 部署 (NPU)

### 6.1 Dockerfile.npu

```dockerfile
# 华为昇腾基础镜像 (含 CANN + torch_npu)
FROM ascendhub.huawei.com/public-ascendhub/ascend-pytorch:24.0.RC1-2.3.1-ubuntu22.04

WORKDIR /workspace/CosyVoice
COPY . .

ENV PYTHONPATH="/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS"

# 安装项目依赖 (排除 NVIDIA 专有包)
RUN grep -v "^tensorrt\|^onnxruntime-gpu\|^deepspeed" requirements.txt > /tmp/req.txt && \
    pip install -r /tmp/req.txt && \
    pip install torchcodec python-multipart

RUN mkdir -p speakers uploads static/wavs logs
EXPOSE 8080

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "1", "--log-level", "info", "--timeout-keep-alive", "300"]
```

### 6.2 docker-compose.npu.yml

```yaml
services:
  cosyvoice-tts:
    image: cosyvoice-tts-api:npu
    container_name: cosyvoice-tts
    restart: unless-stopped

    # NPU 设备映射 (替代 NVIDIA 的 deploy.resources.reservations.devices)
    devices:
      - /dev/davinci0:/dev/davinci0          # NPU 卡 0
      - /dev/davinci_manager:/dev/davinci_manager
      - /dev/devmm_svm:/dev/devmm_svm
      - /dev/hisi_hdc:/dev/hisi_hdc

    volumes:
      - /usr/local/Ascend:/usr/local/Ascend:ro     # CANN 运行时
      - /usr/local/sbin:/usr/local/sbin:ro          # npu-smi 等工具
      - ./pretrained_models:/workspace/CosyVoice/pretrained_models
      - ./speakers:/workspace/CosyVoice/speakers
      - ./app.py:/workspace/CosyVoice/app.py
      # ... 其他挂载同 NVIDIA 版

    environment:
      - ASCEND_RT_VISIBLE_DEVICES=0               # 指定 NPU 卡号
      - MODEL_DIR=pretrained_models/CosyVoice2-0.5B
      - FP16=true
      - LOAD_JIT=false
      - LOAD_TRT=false
      - LOAD_VLLM=false
      - MAX_CONCURRENCY=4

    ports:
      - "8080:8080"

    networks:
      - cosyvoice-net

networks:
  cosyvoice-net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.30.0.0/16
```

---

## 7. 预期性能

| 指标 | A800 80GB (CUDA) | 910B 64GB (NPU) | 说明 |
|------|------------------|------------------|------|
| RTF (充分预热) | 0.13~0.15 | 0.2~0.3 (预估) | NPU 首次适配, 未深度优化 |
| 首次推理延迟 | ~3s | ~5s (预估) | NPU 算子编译较慢 |
| 显存占用 | ~8GB | ~8GB | 模型大小相同 |

> NPU 首次部署性能可能偏低, 主要因为:
> 1. `transfer_to_npu` 的自动映射有开销
> 2. 部分 CosyVoice 算子可能 fallback 到 CPU
> 3. 需要后续用 `npu-smi` 分析瓶颈做针对性优化

---

## 8. 进阶: vLLM-Ascend

如果需要在 NPU 上启用 vLLM 加速:

```bash
# 安装 vLLM-Ascend (华为与社区共同维护的昇腾适配版)
pip install vllm-ascend
```

vLLM-Ascend 保留了 vLLM 原生的 PagedAttention 和 API 接口, 但底层调用 torch_npu + CANN。
目前 CosyVoice 的自定义模型 (CosyVoice2ForCausalLM) 注册到 vLLM-Ascend 的流程与 NVIDIA 版相同。

---

## 9. 验证步骤

```bash
# 1. 检查 NPU 可用性
npu-smi info

# 2. 检查 torch_npu
python -c "
import torch
import torch_npu
print(f'torch: {torch.__version__}')
print(f'torch_npu: {torch_npu.__version__}')
print(f'NPU available: {torch.npu.is_available()}')
print(f'NPU name: {torch.npu.get_device_name(0)}')
"

# 3. 启动服务
docker compose -f docker-compose.npu.yml up -d
docker compose -f docker-compose.npu.yml logs -f

# 4. 测试
curl http://localhost:8080/health
python test_tts.py --host localhost --port 8080
```
