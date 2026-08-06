"""
app.py — Streamlit demo app for Telco Customer Churn classification.

Lets the user:
  1. Upload a test CSV (same schema as test_data.csv, must include the
     target column 'Churn' with Yes/No values for evaluation)
  2. Pick one of the 6 trained models from a dropdown
  3. View Accuracy / AUC / Precision / Recall / F1 / MCC for that model
  4. View the confusion matrix and full classification report
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_DIR = "model"

st.set_page_config(page_title="Telco Churn Classifier Demo", layout="wide")

st.title("📉 Telco Customer Churn — Multi-Model Classifier Demo")
st.write(
    "Upload test data, pick a model, and see how it performs on churn "
    "prediction across six classic ML algorithms."
)


@st.cache_resource
def load_meta():
    with open(os.path.join(MODEL_DIR, "meta.json")) as f:
        return json.load(f)


@st.cache_resource
def load_label_encoder():
    return joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))


@st.cache_resource
def load_model(model_file):
    return joblib.load(os.path.join(MODEL_DIR, model_file))


MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
}

meta = load_meta()
label_encoder = load_label_encoder()
target_col = meta["target_col"]
feature_cols = meta["feature_cols"]

# --- Sidebar controls -------------------------------------------------
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload test data (CSV)", type=["csv"], help="Use test_data.csv or similar."
)

model_choice = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()))

# --- Main logic ---------------------------------------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(data.head())

    missing_cols = [c for c in feature_cols if c not in data.columns]
    if missing_cols:
        st.error(f"Uploaded file is missing expected feature columns: {missing_cols}")
        st.stop()

    has_target = target_col in data.columns

    X = data[feature_cols]

    model_file = MODEL_FILES[model_choice]
    pipe = load_model(model_file)

    y_pred_encoded = pipe.predict(X)
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

    if hasattr(pipe, "predict_proba"):
        y_proba = pipe.predict_proba(X)[:, 1]
    else:
        y_proba = pipe.decision_function(X)

    result_df = data.copy()
    result_df["Predicted_" + target_col] = y_pred_labels
    result_df["Churn_Probability"] = np.round(y_proba, 4)

    st.subheader(f"Predictions — {model_choice}")
    st.dataframe(result_df.head(20))

    if has_target:
        y_true_labels = data[target_col]
        y_true_encoded = label_encoder.transform(y_true_labels)

        acc = accuracy_score(y_true_encoded, y_pred_encoded)
        auc = roc_auc_score(y_true_encoded, y_proba)
        prec = precision_score(y_true_encoded, y_pred_encoded, zero_division=0)
        rec = recall_score(y_true_encoded, y_pred_encoded, zero_division=0)
        f1 = f1_score(y_true_encoded, y_pred_encoded, zero_division=0)
        mcc = matthews_corrcoef(y_true_encoded, y_pred_encoded)

        st.subheader("Evaluation Metrics")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{acc:.3f}")
        m2.metric("AUC", f"{auc:.3f}")
        m3.metric("Precision", f"{prec:.3f}")
        m4.metric("Recall", f"{rec:.3f}")
        m5.metric("F1 Score", f"{f1:.3f}")
        m6.metric("MCC", f"{mcc:.3f}")

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true_encoded, y_pred_encoded)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        with col_b:
            st.subheader("Classification Report")
            report = classification_report(
                y_true_encoded,
                y_pred_encoded,
                target_names=label_encoder.classes_,
                output_dict=True,
                zero_division=0,
            )
            st.dataframe(pd.DataFrame(report).transpose().round(3))
    else:
        st.info(
            "No target column found in the uploaded file — showing predictions "
            "only. Include the 'Churn' column to see evaluation metrics."
        )

    st.subheader("All-Models Comparison (from training run)")
    summary_path = os.path.join(MODEL_DIR, "metrics_summary.csv")
    if os.path.exists(summary_path):
        st.dataframe(pd.read_csv(summary_path).round(4))

else:
    st.info("👈 Upload a CSV file from the sidebar to get started.")
    st.caption(
        "Expected columns: " + ", ".join(feature_cols) + f", and optionally '{target_col}'."
    )
