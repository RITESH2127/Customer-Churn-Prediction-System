import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import uvicorn

app = FastAPI(title="Customer Churn Prediction API")

MODEL_PATH = os.path.join("models", "best_model.joblib")
PIPELINE_PATH = os.path.join("models", "preprocessor.joblib")

# Load model and preprocessor
model = None
preprocessor = None

@app.on_event("startup")
def load_artifacts():
    global model, preprocessor
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PIPELINE_PATH)
        print("Model and preprocessor loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load model or preprocessor: {e}")

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str

@app.post("/predict")
def predict_churn(data: CustomerData):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    try:
        # Convert incoming data to dataframe
        df = pd.DataFrame([data.dict()])
        
        # Consistent preprocessing logic
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
        # Transform data
        X_processed = preprocessor.transform(df)
        
        # Predict
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0]
        
        # The positive class (1) corresponds to "Churn"
        churn_prob = float(probability[1])
        is_churn = bool(prediction == 1)
        
        return {
            "prediction": "Churn" if is_churn else "Retained",
            "probability": churn_prob
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Mount frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join("frontend", "index.html"))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
