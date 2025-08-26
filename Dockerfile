# Pro-mode worker: faster-whisper (ASR) + WhisperX alignment
# База такая же, как у runpod worker-faster_whisper (cudnn9)
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    WHISPER_CACHE=/root/.cache/whisper \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

# Базовые зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev ffmpeg libsndfile1 git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Pip обновим
RUN python3 -m pip install --upgrade pip

# ✅ Torch/Torchaudio с CUDA **12.4** (совместимо с cuDNN 9 в базе)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch==2.4.0+cu124 torchaudio==2.4.0+cu124

# CTranslate2 (GPU) + faster-whisper
# (диапазон CTranslate2 устойчив под CUDA12; при желании зафиксируй ровно 4.4.0)
RUN pip install --no-cache-dir 'ctranslate2>=4.4,<5' 'faster-whisper>=1.0,<2'

# WhisperX, RunPod SDK и утилиты
RUN pip install --no-cache-dir whisperx==3.4.2 runpod==1.7.13 requests srt numpy

WORKDIR /src
COPY handler.py /src/handler.py

# Санчек синтаксиса хэндлера
RUN python3 -m py_compile /src/handler.py

CMD ["python3", "-u", "handler.py"]
