from fastapi import FastAPI
import pickle
import pandas as pd
from ml.train_model import get_model_metrics


app = FastAPI()

# Load model & scaler
model = pickle.load(open("ml/models/model.pkl", "rb"))
scaler = pickle.load(open("ml/models/scaler.pkl", "rb"))



@app.get("/")
def home():
    return {"message": "Loan Risk Prediction API is running"}


# @app.post("/predict")
# def predict(data: dict):
    
#     # Convert input to DataFrame
#     df = pd.DataFrame([data])

#     # Scale input
#     df_scaled = scaler.transform(df)

#     # Predict
#     prediction = model.predict(df_scaled)[0]
#     probability = model.predict_proba(df_scaled)[0][1]

#     return {
#         "prediction": int(prediction),
#         "probability": float(probability)
#     }

# @app.post("/predict")
# def predict(data: dict):
    
#     df = pd.DataFrame([data])
#     df_scaled = scaler.transform(df)

#     prediction = model.predict(df_scaled)[0]
#     probability = model.predict_proba(df_scaled)[0][1]

#     # 🔥 Convert to human-readable
#     if prediction == 1:
#         message = "Loan Approved ✅"
#     else:
#         message = "Loan Rejected ❌"


#     return {
#         "message": message,
#         "probability": round(float(probability), 2)
#     }
@app.post("/predict")
def predict(data: dict):
    
    df = pd.DataFrame([data])
    df_scaled = scaler.transform(df)

    # Prediction
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    # Message
    if prediction == 1:
        message = "Loan Approved ✅"
    else:
        message = "Loan Rejected ❌"

    # 🔥 Get model metrics
    metrics = get_model_metrics()

    # ✅ Final response
    return {
        "prediction": {
            "message": message,
            "probability": round(float(probability), 2)
        },
        "model_metrics": metrics
    }