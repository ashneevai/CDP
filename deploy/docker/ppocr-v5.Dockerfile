FROM python:3.11-slim

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY packages ./packages
COPY apps ./apps
COPY workers ./workers
COPY config ./config
RUN pip install --no-cache-dir . \
    "rapidocr-onnxruntime>=1.3,<2" "onnxruntime>=1.17,<2" \
    "paddlepaddle>=3,<4" "paddleocr>=3,<4"

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /data /models && chown -R appuser:appuser /data /models
USER appuser
CMD ["python", "-m", "workers.retry.consumer"]
