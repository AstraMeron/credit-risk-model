from fastapi import FastAPI, HTTPException
import mlflow.sklearn
import pandas as pd
from src.api.pydantic_models import PredictionRequest, PredictionResponse
import os

app = FastAPI(title="Credit Risk Prediction API")

# Path to the local MLflow DB we created in Task 5
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load the best model from the registry
# We use the name you gave it in Task 5
MODEL_NAME = "Credit_Risk_Model"
MODEL_STAGE = "None" # Or "Production" if you set it in the UI

try:
    model_uri = f"models:/{MODEL_NAME}/latest"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Model {MODEL_NAME} loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/")
def home():
    return {"message": "Credit Risk API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Convert input data to DataFrame for the model
    input_df = pd.DataFrame([data.dict()])
    
    # Get prediction and probability
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])
    
    return {
        "risk_probability": probability,
        "prediction": prediction
    }