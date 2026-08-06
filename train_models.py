"""
train_models.py
----------------
Trains 6 classification models on the Telco Customer Churn dataset,
evaluates them with Accuracy, AUC, Precision, Recall, F1, and MCC,
and saves:
  - trained models + preprocessing objects (model/*.pkl)
  - a held-out test set as CSV (test_data.csv) for the Streamlit app
  - a metrics comparison table (model/metrics_summary.csv)

Usage:
    python train_models.py --data telco_customer_churn.csv
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

MODEL_DIR = "model"
RANDOM_STATE = 42


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Standard Telco Churn quirks: TotalCharges is sometimes read as object
    # because of blank strings for customers with 0 tenure.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df = df.dropna()
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE
        ),
    }


def evaluate(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main(args):
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_and_clean(args.data)

    target_col = args.target
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Columns: {list(df.columns)}"
        )

    y_raw = df[target_col]
    X = df.drop(columns=[target_col])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)  # e.g. Yes/No -> 1/0
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Save the raw (unscaled) test split + true labels as test_data.csv
    # This is what gets uploaded to the Streamlit app.
    test_export = X_test.copy()
    test_export[target_col] = label_encoder.inverse_transform(y_test)
    test_export.to_csv("test_data.csv", index=False)
    print(f"Saved test_data.csv with shape {test_export.shape}")

    results = []
    models = get_models()

    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        else:
            y_proba = pipe.decision_function(X_test)

        metrics = evaluate(y_test, y_pred, y_proba)
        metrics["Model"] = name
        results.append(metrics)

        fname = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(pipe, os.path.join(MODEL_DIR, fname))
        print(f"Trained {name}: {metrics}")

    results_df = pd.DataFrame(results)[
        ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]
    results_df.to_csv(os.path.join(MODEL_DIR, "metrics_summary.csv"), index=False)

    # Save column metadata so the Streamlit app knows how to align uploaded CSVs
    meta = {
        "target_col": target_col,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_cols": numeric_cols + categorical_cols,
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n=== Final Comparison Table ===")
    print(results_df.round(4).to_string(index=False))
    print("\nAll models, metrics, and metadata saved to ./model/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, required=True, help="Path to the Telco churn CSV file"
    )
    parser.add_argument(
        "--target", type=str, default="Churn", help="Name of the target column"
    )
    args = parser.parse_args()
    main(args)
