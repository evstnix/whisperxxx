# faster-whisper turbo требует cuDNN >= 9 — поэтому у runpod-воркера база с cudnn9.
# Мы делаем WhisperX-воркер на той же базе.
# ref: runpod-workers/worker-faster_whisper Dockerfile
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive \
PIP_DISABLE_PIP_VERSION_CHECK=1 \
HF_HOME=/root/.cache/huggingface \
WHISPER_CACHE=/root/.cache/whisper \
TOKENIZERS_PARALLELISM=false \
PYTHONUNBUFFERED=1


# Базовые зависимости: Python 3.10, pip, ffmpeg, sndfile, git, сертификаты
RUN apt-get update && apt-get install -y --no-install-recommends \
python3 python3-pip python3-dev ffmpeg libsndfile1 git ca-certificates && \
rm -rf /var/lib/apt/lists/*


# Обновим pip
RUN python3 -m pip install --upgrade pip


# Установка PyTorch с CUDA-колёсами (cu121) + torchaudio.
# Эти колёса включают нужные CUDA/cuDNN библиотеки внутри себя.
# Если поймаешь несовместимость — см. комментарий ниже про cu124.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
torch==2.1.0+cu121 torchaudio==2.1.0+cu121


# CTranslate2 (GPU) + faster-whisper
# Диапазон версий CTranslate2 подобран под CUDA12.x; при редких багах можно зафиксировать 4.4.0
RUN pip install 'ctranslate2>=4.4,<5' faster-whisper


# WhisperX, RunPod SDK и утилиты
RUN pip install whisperx==3.4.2 runpod==1.7.13 requests srt numpy


WORKDIR /src
COPY handler.py /src/handler.py


CMD ["python3", "-u", "handler.py"]


# ---
# Если потребуется перейти на CUDA 12.4 стэк PyTorch (для новых базовых образов):
# просто замените блок установки torch на:
# RUN pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
# torch==2.4.0+cu124 torchaudio==2.4.0+cu124
# ---