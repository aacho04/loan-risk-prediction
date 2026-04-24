import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("data/processed/cleaned.csv")

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("ml/models/model.pkl", "wb"))

# Save scaler
pickle.dump(scaler, open("ml/models/scaler.pkl", "wb"))

print("✅ Model and Scaler saved")