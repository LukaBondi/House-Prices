"""Train the baseline XGB pipeline and save it under models/."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "train.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "baseline_xgb_pipeline.pkl"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
RANDOM_STATE = 42


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    target = "SalePrice"
    features = [c for c in df.columns if c not in ["Id", target]]
    X = df[features]
    y = df[target]

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = build_pipeline(numeric_features, categorical_features)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = r2_score(y_test, preds)
    rmsle = float(
        np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(np.clip(preds, 0, None))))
    )

    print(f"MAE   : {mae:,.2f}")
    print(f"RMSE  : {rmse:,.2f}")
    print(f"R2    : {r2:.4f}")
    print(f"RMSLE : {rmsle:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    import json

    schema = {
        "features": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "sample": X_test.head(1).replace({np.nan: None}).to_dict(orient="records")[0],
    }
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved schema to {SCHEMA_PATH}")


if __name__ == "__main__":
    main()
