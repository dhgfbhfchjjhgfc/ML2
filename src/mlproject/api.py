"""FastAPI сервис для антифрод-скоринга транзакций (V20)."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
_model = None

class Features(BaseModel):
    Amount: float = Field(..., ge=0.0, le=30000.0, description="Сумма транзакции")
    V1: float = Field(..., ge=-60.0, le=10.0)
    V2: float = Field(..., ge=-80.0, le=30.0)
    V3: float = Field(..., ge=-50.0, le=10.0)
    V4: float = Field(..., ge=-10.0, le=20.0)
    V5: float = Field(..., ge=-120.0, le=40.0)
    V6: float = Field(..., ge=-30.0, le=80.0)
    V7: float = Field(..., ge=-50.0, le=130.0)
    V8: float = Field(..., ge=-80.0, le=30.0)
    V9: float = Field(..., ge=-20.0, le=20.0)
    V10: float = Field(..., ge=-30.0, le=30.0)
    V11: float = Field(..., ge=-10.0, le=20.0)
    V12: float = Field(..., ge=-20.0, le=10.0)
    V13: float = Field(..., ge=-10.0, le=10.0)
    V14: float = Field(..., ge=-25.0, le=15.0)
    V15: float = Field(..., ge=-10.0, le=10.0)
    V16: float = Field(..., ge=-30.0, le=20.0)
    V17: float = Field(..., ge=-30.0, le=15.0)
    V18: float = Field(..., ge=-15.0, le=10.0)
    V19: float = Field(..., ge=-10.0, le=10.0)
    V20: float = Field(..., ge=-60.0, le=40.0)
    V21: float = Field(..., ge=-40.0, le=35.0)
    V22: float = Field(..., ge=-15.0, le=15.0)
    V23: float = Field(..., ge=-50.0, le=30.0)
    V24: float = Field(..., ge=-10.0, le=10.0)
    V25: float = Field(..., ge=-15.0, le=15.0)
    V26: float = Field(..., ge=-10.0, le=10.0)
    V27: float = Field(..., ge=-30.0, le=35.0)
    V28: float = Field(..., ge=-20.0, le=40.0)

    model_config = {"extra": "forbid"}

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Файл модели не найден по пути: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/predict")
def predict(features: Features) -> dict[str, Any]:
    model = get_model()
    df = pd.DataFrame([features.model_dump()])
    proba = float(model.predict_proba(df)[0, 1])
    pred = int(proba >= 0.5)
    return {
        "prediction": pred,
        "probability": round(proba, 4),
        "model_version": "1.0.0"
    }
