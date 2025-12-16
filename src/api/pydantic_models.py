from pydantic import BaseModel

class PredictionRequest(BaseModel):
    Amount: float
    Value: float
    PricingStrategy: int

class PredictionResponse(BaseModel):
    risk_probability: float
    prediction: int