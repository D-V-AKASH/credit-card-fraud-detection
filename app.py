from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')
THRESHOLD = 0.6

@app.get("/")
def root():
    return {"status": "Fraud detection API is running"}

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    df_scaled = df.copy()
    df_scaled[['Amount', 'Time']] = scaler.transform(df_scaled[['Amount', 'Time']])
    df_scaled = df_scaled[feature_names]

    probs = model.predict_proba(df_scaled)[:, 1]
    preds = (probs >= THRESHOLD).astype(bool)

    results = []
    for i in range(len(df)):
        results.append({
            "row": i + 1,
            "fraud_probability": round(float(probs[i]), 4),
            "is_fraud": bool(preds[i])
        })

    return {"count": len(results), "predictions": results}