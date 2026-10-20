FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY model_training/requirements-trainer.txt /app/model_training/requirements-trainer.txt
RUN pip install --no-cache-dir -r /app/model_training/requirements-trainer.txt

COPY backend /app/backend
COPY model_training /app/model_training
COPY dataset/train.csv /app/dataset/train.csv
COPY dataset/test.csv /app/dataset/test.csv
COPY dataset/split_manifest.json /app/dataset/split_manifest.json
COPY models/baseline_xgb_pipeline.pkl /app/models/baseline_xgb_pipeline.pkl
COPY models/house_prices_native.zip /app/models/house_prices_native.zip
COPY models/feature_schema.json /app/models/feature_schema.json
COPY models/drift_reference.json /app/models/drift_reference.json

ENV DATA_DIR=/data \
    DB_PATH=/data/requests.sqlite3 \
    MODEL_REGISTRY_PATH=/artifacts \
    MODEL_STATE_PATH=/state \
    DRIFT_REFERENCE_PATH=/app/models/drift_reference.json \
    PYTHONUNBUFFERED=1

CMD ["python", "-m", "model_training.retrain", "--interval", "3600"]
