import pandas as pd

# 1. Load cleaned data
df = pd.read_csv("data/processed/cleaned.csv")

print("✅ Data Loaded for Training")


# 2. Split into features & target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]


# 3. Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Import models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# 5. Initialize models
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier()
xgb = XGBClassifier(eval_metric='logloss')


# 6. Train models
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)


# 7. Predictions
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)
xgb_pred = xgb.predict(X_test)


# 8. Evaluation
from sklearn.metrics import accuracy_score, f1_score

print("\n📊 Model Performance:\n")

print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("F1 Score:", f1_score(y_test, lr_pred))

print("\nRandom Forest:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("F1 Score:", f1_score(y_test, rf_pred))

print("\nXGBoost:")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("F1 Score:", f1_score(y_test, xgb_pred))