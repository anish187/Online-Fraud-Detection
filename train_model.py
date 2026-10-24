"""Train and evaluate the educational Online Fraud Detection model."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "online_fraud_detection_dataset.csv"
MODEL_PATH = BASE_DIR / "fraud_model.joblib"
FEATURES = [
    "transaction_amount",
    "transaction_hour",
    "new_device",
    "international_transaction",
    "failed_attempts",
    "account_age_days",
]
TARGET = "is_fraud"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    # Keep only required columns, convert to numeric, and reject unusable rows.
    data = df[required].copy()
    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    before = len(data)
    data = data.dropna(subset=required)
    data = data[data[TARGET].isin([0, 1])]
    data = data[(data["transaction_amount"] >= 0)
                & data["transaction_hour"].between(0, 23)
                & data["new_device"].isin([0, 1])
                & data["international_transaction"].isin([0, 1])
                & (data["failed_attempts"] >= 0)
                & (data["account_age_days"] >= 0)]
    data[TARGET] = data[TARGET].astype(int)
    if data[TARGET].nunique() != 2:
        raise ValueError("The dataset must contain both target classes: 0 and 1.")

    print("DATASET CHECK")
    print("Original rows:", before)
    print("Rows used:", len(data))
    print("Missing values in required columns before cleaning:")
    print(df[required].isna().sum().to_string())
    print("Target counts (0=legitimate label, 1=fraud label):")
    print(data[TARGET].value_counts().sort_index().to_string())

    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # A pipeline keeps the expected feature order together with the model.
    model = Pipeline([
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
        ))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nHELD-OUT TEST RESULTS (synthetic educational data only)")
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_test, predictions, zero_division=0):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, probabilities):.4f}")
    print("\nConfusion matrix (rows=actual [0,1], columns=predicted [0,1]):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=[0, 1],
                                target_names=["Legitimate label (0)", "Fraud label (1)"],
                                zero_division=0))

    # Save the fitted pipeline and feature metadata for both interfaces.
    joblib.dump({"model": model, "features": FEATURES, "target": TARGET}, MODEL_PATH)
    print(f"\nSaved trained model to: {MODEL_PATH}")
    print("Reminder: these metrics describe this synthetic dataset, not real banking performance.")


if __name__ == "__main__":
    main()
