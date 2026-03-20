# ============================================================
# CosyVoice TTS API - 生产 Docker 镜像
# 目标 GPU: NVIDIA L20 (48GB) / A800 (80GB)
# 基础镜像: CUDA 12.4 + cuDNN (devel 版本含编译工具)
# ============================================================

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG VENV_NAME="cosyvoice"
ENV VENV=$VENV_NAME
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "--login", "-c"]

# ── 系统依赖 ──
RUN apt-get update -y --fix-missing && \
    apt-get install -y \
        git build-essential curl wget \
        ffmpeg sox libsox-dev \
        unzip git-lfs ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    git lfs install

# ── Miniforge (conda) ──
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
    /bin/bash ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> /opt/nvidia/entrypoint.d/100.conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate ${VENV}" >> /opt/nvidia/entrypoint.d/110.conda_default_env.sh && \
    echo "conda activate ${VENV}" >> $HOME/.bashrc
ENV PATH /opt/conda/bin:$PATH
RUN conda config --add channels conda-forge && conda config --set channel_priority strict

# ── Python 3.10 环境 ──
RUN conda create -y -n ${VENV} python=3.10
ENV CONDA_DEFAULT_ENV=${VENV}
ENV PATH /opt/conda/bin:/opt/conda/envs/${VENV}/bin:$PATH
RUN conda activate ${VENV} && conda install -y -c conda-forge pynini==2.1.5

WORKDIR /workspace

# ── 复制本地项目代码 (docker build context = 项目根目录) ──
COPY . /workspace/CosyVoice
ENV PYTHONPATH="${PYTHONPATH}:/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS"

# ── 安装项目依赖 (requirements.txt) ──
RUN conda activate ${VENV} && cd /workspace/CosyVoice && \
    pip install -r requirements.txt \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        --trusted-host=mirrors.aliyun.com \
        --no-cache-dir

# ── 安装 API 服务额外依赖 ──
# torchcodec: CosyVoice 内部 load_wav / add_zero_shot_spk 需要
# python-multipart: FastAPI 文件上传
RUN conda activate ${VENV} && \
    pip install torchcodec python-multipart>=0.0.6 \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        --trusted-host=mirrors.aliyun.com \
        --no-cache-dir

# ── (可选) 安装 vLLM: LLM decoding 加速 4x ──
# 取消下面注释即可启用, 注意 vLLM 对 torch 版本有严格要求
# 安装后设置环境变量 LOAD_VLLM=true
# RUN conda activate ${VENV} && \
#     pip install vllm==v0.9.0 transformers==4.51.3 numpy==1.26.4 \
#         -i https://mirrors.aliyun.com/pypi/simple/ \
#         --trusted-host=mirrors.aliyun.com \
#         --no-cache-dir

# ── 创建运行时目录 ──
RUN mkdir -p /workspace/CosyVoice/speakers \
             /workspace/CosyVoice/uploads \
             /workspace/CosyVoice/static/wavs \
             /workspace/CosyVoice/logs

WORKDIR /workspace/CosyVoice
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=180s \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate cosyvoice && python -m uvicorn app:app --host 0.0.0.0 --port 8080 --workers 1 --log-level info --timeout-keep-alive 300"]
