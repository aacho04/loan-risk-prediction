import pickle
import pandas as pd

# 1. Load model & scaler
model = pickle.load(open("ml/models/model.pkl", "rb"))
scaler = pickle.load(open("ml/models/scaler.pkl", "rb"))

print("✅ Model loaded")

# 2. Create sample input (MUST match your training columns)
input_data = {
    "Dependents": 1,
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 2000,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Total_Income": 7000,
    "Debt_to_Income": 150/7000,
    "Gender_Male": 1,
    "Married_Yes": 1,
    "Education_Not Graduate": 0,
    "Self_Employed_Yes": 0,
    "Property_Area_Semiurban": 1,
    "Property_Area_Urban": 0
}

# 3. Convert to DataFrame
df = pd.DataFrame([input_data])

# 4. Scale input
df_scaled = scaler.transform(df)

# 5. Predict
prediction = model.predict(df_scaled)[0]
probability = model.predict_proba(df_scaled)[0][1]

# 6. Show result
if prediction == 1:
    print("✅ Loan Approved")
else:
    print("❌ Loan Rejected")

print(f"📊 Probability of approval: {probability:.2f}")