import pandas as pd

def get_model_metrics():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, f1_score

    # 1. Load data
    df = pd.read_csv("data/processed/cleaned.csv")

    # 2. Split
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Models
    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier()
    xgb = XGBClassifier(eval_metric='logloss')

    # 5. Train
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    # 6. Predictions
    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)
    xgb_pred = xgb.predict(X_test)

    # 7. RETURN JSON (IMPORTANT)
    return {
        "Logistic Regression": {
            "accuracy": round(accuracy_score(y_test, lr_pred), 2),
            "f1_score": round(f1_score(y_test, lr_pred), 2)
        },
        "Random Forest": {
            "accuracy": round(accuracy_score(y_test, rf_pred), 2),
            "f1_score": round(f1_score(y_test, rf_pred), 2)
        },
        "XGBoost": {
            "accuracy": round(accuracy_score(y_test, xgb_pred), 2),
            "f1_score": round(f1_score(y_test, xgb_pred), 2)
        }
    }