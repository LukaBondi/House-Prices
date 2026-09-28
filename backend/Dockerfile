FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements-runtime.txt /app/backend/requirements-runtime.txt
RUN pip install --no-cache-dir -r /app/backend/requirements-runtime.txt

COPY backend /app/backend
COPY models/house_prices_native.zip /app/models/house_prices_native.zip
COPY models/feature_schema.json /app/models/feature_schema.json

ENV MODEL_PATH=/app/models/house_prices_native.zip
ENV SCHEMA_PATH=/app/models/feature_schema.json
ENV WEB_CONCURRENCY=1
ENV PREDICTION_CACHE_ENABLED=1
ENV PREDICTION_CACHE_MAX_ENTRIES=1024
ENV PREDICTION_CACHE_TTL_SECONDS=300
ENV DYNAMIC_BATCHING_ENABLED=0
ENV BATCH_MAX_SIZE=16
ENV BATCH_MAX_DELAY_MS=0
ENV BATCH_QUEUE_MAX_SIZE=256
ENV INFERENCE_WORKERS=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY:-1}"]
