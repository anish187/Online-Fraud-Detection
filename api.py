"""Optional FastAPI endpoint. Start after training fraud_model.joblib."""
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fraud_model.joblib"
app = FastAPI(title="Online Fraud Detection API", version="1.0.0",
              description="Educational API using a synthetic-data Random Forest model.")

class Transaction(BaseModel):
    transaction_amount: float = Field(ge=0, description="Transaction amount")
    transaction_hour: int = Field(ge=0, le=23)
    new_device: int = Field(ge=0, le=1, description="0=no, 1=yes")
    international_transaction: int = Field(ge=0, le=1, description="0=no, 1=yes")
    failed_attempts: int = Field(ge=0)
    account_age_days: int = Field(ge=0)

@app.get("/")
def home():
    return {"message": "Online Fraud Detection API is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}

@app.post("/predict")
def predict(transaction: Transaction):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model file missing. Run python train_model.py first.")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = bundle["features"]
    row = pd.DataFrame([transaction.model_dump()], columns=features)
    label = int(model.predict(row)[0])
    score = float(model.predict_proba(row)[0][1])
    return {
        "prediction": label,
        "label": "potentially_fraudulent" if label == 1 else "legitimate",
        "model_estimated_fraud_score": round(score, 4),
        "warning": "Educational synthetic-data model only; not suitable for real financial decisions."
    }
