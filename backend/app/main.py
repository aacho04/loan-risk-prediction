from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# Load model & scaler
model = pickle.load(open("ml/models/model.pkl", "rb"))
scaler = pickle.load(open("ml/models/scaler.pkl", "rb"))


@app.get("/")
def home():
    return {"message": "Loan Risk Prediction API is running"}


@app.post("/predict")
def predict(data: dict):
    
    # Convert input to DataFrame
    df = pd.DataFrame([data])

    # Scale input
    df_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }