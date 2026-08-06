"""
app.py — Streamlit demo app for Telco Customer Churn classification.

Lets the user:
  1. Upload a test CSV (same schema as test_data.csv, must include the
     target column 'Churn' with Yes/No values for evaluation)
  2. Pick one of the 6 trained models from a dropdown
  3. View Accuracy / AUC / Precision / Recall / F1 / MCC for that model,
     styled as colored metric cards
  4. Explore an interactive confusion matrix, ROC curve, classification
     report, feature importance (tree-based models), and a per-model
     comparison chart across all six algorithms
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    roc_curve,
)

MODEL_DIR = "model"

st.set_page_config(
    page_title="Telco Churn Classifier",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Light custom styling — colored metric cards, tighter spacing, accent color
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border-radius: 12px;
        padding: 18px 12px;
        text-align: center;
        border: 1px solid #2d3748;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.8rem;
        color: #9ca3af;
        font-weight: 500;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .metric-card p {
        margin: 4px 0 0 0;
        font-size: 1.6rem;
        font-weight: 700;
        color: #f9fafb;
    }
    .risk-badge-high {
        background-color: #fee2e2; color: #991b1b;
        padding: 3px 10px; border-radius: 999px; font-weight: 600; font-size: 0.8rem;
    }
    .risk-badge-low {
        background-color: #dcfce7; color: #166534;
        padding: 3px 10px; border-radius: 999px; font-weight: 600; font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📉 Telco Customer Churn — Multi-Model Classifier")
st.caption(
    "Upload test data, pick a model, and explore churn predictions across "
    "six classic ML algorithms — with live metrics, confusion matrix, ROC "
    "curve, and feature importance."
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


@st.cache_data
def load_summary():
    path = os.path.join(MODEL_DIR, "metrics_summary.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


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
summary_df = load_summary()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader(
        "Upload test data (CSV)", type=["csv"], help="Use test_data.csv or similar."
    )
    model_choice = st.selectbox("Select a model", list(MODEL_FILES.keys()))

    st.divider()
    st.markdown("**About**")
    st.caption(
        "Six models trained on the Telco Customer Churn dataset: "
        "Logistic Regression, Decision Tree, kNN, Naive Bayes, and a "
        "Random Forest ensemble — each tuned via GridSearchCV."
    )

    if summary_df is not None:
        best_row = summary_df.loc[summary_df["MCC"].idxmax()]
        st.success(f"🏆 Best MCC: **{best_row['Model']}** ({best_row['MCC']:.3f})")


def metric_card(col, label, value):
    col.markdown(
        f"""<div class="metric-card"><h3>{label}</h3><p>{value}</p></div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

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

    tab_overview, tab_predictions, tab_diagnostics, tab_compare = st.tabs(
        ["📊 Overview", "🔮 Predictions", "🩺 Model Diagnostics", "⚖️ Compare All Models"]
    )

    # ---------------------- Overview tab ---------------------------------
    with tab_overview:
        st.subheader(f"Snapshot — {model_choice}")

        churn_count = (y_pred_labels == "Yes").sum() if "Yes" in label_encoder.classes_ else int(y_pred_encoded.sum())
        churn_rate = churn_count / len(y_pred_labels) * 100

        c1, c2, c3 = st.columns(3)
        metric_card(c1, "Customers Scored", f"{len(data):,}")
        metric_card(c2, "Predicted Churners", f"{churn_count:,}")
        metric_card(c3, "Predicted Churn Rate", f"{churn_rate:.1f}%")

        st.markdown("")
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=churn_rate,
                title={"text": "Average Predicted Churn Risk (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#dc2626"},
                    "steps": [
                        {"range": [0, 30], "color": "#dcfce7"},
                        {"range": [30, 60], "color": "#fef9c3"},
                        {"range": [60, 100], "color": "#fee2e2"},
                    ],
                },
            )
        )
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.subheader("Preview of Uploaded Data")
        st.dataframe(data.head(10), use_container_width=True)

    # ---------------------- Predictions tab -------------------------------
    with tab_predictions:
        st.subheader(f"Predictions — {model_choice}")

        def highlight_risk(val):
            if isinstance(val, (int, float)) and val >= 0.5:
                return "background-color: #fee2e2; color: #991b1b;"
            return ""

        styled = result_df.head(50).style.applymap(
            highlight_risk, subset=["Churn_Probability"]
        )
        st.dataframe(styled, use_container_width=True)

        st.download_button(
            "⬇️ Download full predictions as CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv",
        )

        fig_hist = px.histogram(
            result_df,
            x="Churn_Probability",
            nbins=25,
            title="Distribution of Predicted Churn Probability",
            color_discrete_sequence=["#3b82f6"],
        )
        fig_hist.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    # ---------------------- Diagnostics tab -------------------------------
    with tab_diagnostics:
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
            cols = st.columns(6)
            for col, label, val in zip(
                cols,
                ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
                [acc, auc, prec, rec, f1, mcc],
            ):
                metric_card(col, label, f"{val:.3f}")

            st.markdown("")
            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_true_encoded, y_pred_encoded)
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    x=list(label_encoder.classes_),
                    y=list(label_encoder.classes_),
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                )
                fig_cm.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_cm, use_container_width=True)

            with col_b:
                st.subheader("ROC Curve")
                fpr, tpr, _ = roc_curve(y_true_encoded, y_proba)
                fig_roc = go.Figure()
                fig_roc.add_trace(
                    go.Scatter(
                        x=fpr, y=tpr, mode="lines",
                        name=f"{model_choice} (AUC={auc:.3f})",
                        line=dict(color="#3b82f6", width=3),
                    )
                )
                fig_roc.add_trace(
                    go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines",
                        name="Random guess", line=dict(color="gray", dash="dash"),
                    )
                )
                fig_roc.update_layout(
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    height=380,
                    margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(x=0.4, y=0.05),
                )
                st.plotly_chart(fig_roc, use_container_width=True)

            st.subheader("Classification Report")
            report = classification_report(
                y_true_encoded,
                y_pred_encoded,
                target_names=label_encoder.classes_,
                output_dict=True,
                zero_division=0,
            )
            report_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(
                report_df.style.background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"]),
                use_container_width=True,
            )

            # Feature importance for tree-based models
            classifier = pipe.named_steps.get("classifier")
            if classifier is not None and hasattr(classifier, "feature_importances_"):
                st.subheader("Feature Importance")
                try:
                    ohe_names = pipe.named_steps["preprocessor"].get_feature_names_out()
                    importances = classifier.feature_importances_
                    imp_df = (
                        pd.DataFrame({"Feature": ohe_names, "Importance": importances})
                        .sort_values("Importance", ascending=False)
                        .head(15)
                    )
                    fig_imp = px.bar(
                        imp_df.sort_values("Importance"),
                        x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale="Blues",
                    )
                    fig_imp.update_layout(height=450, margin=dict(l=10, r=10, t=20, b=10))
                    st.plotly_chart(fig_imp, use_container_width=True)
                except Exception:
                    st.caption("Feature importance unavailable for this pipeline configuration.")
        else:
            st.info(
                "No target column found in the uploaded file — diagnostics need "
                f"ground-truth labels. Include the '{target_col}' column to see "
                "metrics, confusion matrix, and ROC curve here."
            )

    # ---------------------- Compare All Models tab -------------------------
    with tab_compare:
        st.subheader("All Models — Training-Time Comparison")
        if summary_df is not None:
            metric_pick = st.radio(
                "Metric to visualize",
                ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
                horizontal=True,
            )
            fig_bar = px.bar(
                summary_df.sort_values(metric_pick, ascending=True),
                x=metric_pick, y="Model", orientation="h",
                color=metric_pick, color_continuous_scale="Blues",
                text_auto=".3f",
            )
            fig_bar.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("**Full metrics table**")
            st.dataframe(
                summary_df.set_index("Model").style.background_gradient(
                    cmap="Blues", axis=0
                ).format("{:.4f}"),
                use_container_width=True,
            )

            fig_radar = go.Figure()
            metrics_list = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
            for _, row in summary_df.iterrows():
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=[row[m] for m in metrics_list],
                        theta=metrics_list,
                        fill="toself",
                        name=row["Model"],
                        opacity=0.6,
                    )
                )
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                height=500,
                title="Model Profile Comparison (Radar)",
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.warning("No metrics_summary.csv found in model/ — run train_models.py first.")

else:
    st.info("👈 Upload a CSV file from the sidebar to get started.")
    st.caption(
        "Expected columns: " + ", ".join(feature_cols) + f", and optionally '{target_col}'."
    )
    if summary_df is not None:
        st.subheader("Training-Time Model Comparison")
        st.dataframe(
            summary_df.set_index("Model").style.background_gradient(cmap="Blues").format("{:.4f}"),
            use_container_width=True,
        )