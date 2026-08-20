# Akio Studio — RunPod Serverless worker for the video stage.
#
# Build and push:
#   docker build -t <registry>/akio-video-worker:1.0.0 runpod_worker/
#   docker push   <registry>/akio-video-worker:1.0.0
#
# Model weights are NOT baked in: a 14B checkpoint would make a ~60 GB image
# and a brutal cold start. Attach a RunPod network volume and point
# AKIO_MODEL_CACHE at it (default /runpod-volume/models).
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    AKIO_MODEL_CACHE=/runpod-volume/models

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
