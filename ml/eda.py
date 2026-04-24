import pandas as pd

# 1. Load dataset
df = pd.read_csv("data/raw/train.csv")

print("✅ Data Loaded")
print(df.head())


# 2. Drop unnecessary column
df.drop("Loan_ID", axis=1, inplace=True)


# 3. Convert target column
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})


# 4. Fix Dependents column
df["Dependents"] = df["Dependents"].replace("3+", 3)
df["Dependents"] = df["Dependents"].astype(float)


# 5. Handle missing values

# Categorical columns
df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
df["Married"].fillna(df["Married"].mode()[0], inplace=True)
df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

# Numerical columns
df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median(), inplace=True)


# 6. Feature Engineering
df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["Debt_to_Income"] = df["LoanAmount"] / df["Total_Income"]


# 7. Convert categorical → numeric
df = pd.get_dummies(df, drop_first=True)


# 8. Final check
print("\n🔍 Missing values after cleaning:")
print(df.isnull().sum())


# 9. Save cleaned data
df.to_csv("data/processed/cleaned.csv", index=False)

print("\n✅ Cleaned data saved at data/processed/cleaned.csv")