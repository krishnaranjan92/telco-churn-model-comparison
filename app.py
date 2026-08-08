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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

st.set_page_config(
    page_title="Telcom Churn Classifier",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    /* Metric Cards */
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

    /* Soft Gradient Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f7ff 0%, #e0f2fe 100%);
    }

    /* Winner Card Callout */
    .winner-card-light {
        background-color: #ffffff;
        border-left: 5px solid #2563eb;
        border-radius: 8px;
        padding: 14px;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.12);
        margin-top: 10px;
    }
    .winner-header-light {
        color: #2563eb;
        font-weight: 800;
        font-size: 0.9rem;
        text-transform: uppercase;
    }
    .winner-title-light {
        color: #1e40af;
        font-weight: 800;
        font-size: 1.25rem;
        margin: 4px 0;
    }
    .winner-stats-light {
        color: #475569;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(col, title, value):
    col.markdown(
        f"""
        <div class="metric-card">
            <h3>{title}</h3>
            <p>{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def highlight_risk(val):
    if isinstance(val, (int, float)) and val >= 0.5:
        return "background-color: #fee2e2; color: #991b1b;"
    return ""


with st.container(border=True):
    st.title("📊 Telcom Customer Churn — Multi-Model Comparison")
    st.markdown(
        """
    * **Upload Test Data:** Import your evaluation dataset in the sidebar.
    * **Select a Model:** Choose one of the classic machine learning algorithms to evaluate.
    * **Overview:** Explore predictions snapshot for the selected model on the uploaded test dataset.
    * **Prediction:** View model prediction performance and distribution of predictions. You can download the prediction file as well.
    * **Model Diagnostics:** Visualize evaluation metrics, confusion matrix, classification report, and ROC curve.
    * **Compare All Models:** Compare all model scores by metric type and select metrics to visualize performance results.
    """
    )

st.divider()


@st.cache_resource
def load_meta():
    path = os.path.join(MODEL_DIR, "meta.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_resource
def load_label_encoder():
    path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_resource
def load_model(model_file):
    path = os.path.join(MODEL_DIR, model_file)
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_data
def load_summary():
    path = os.path.join(MODEL_DIR, "metrics_summary.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data
def load_ranked_summary():
    path = os.path.join(MODEL_DIR, "ranked_summary.csv")
    if os.path.exists(path):
        return pd.read_csv(path)

    df = load_summary()
    if df is None:
        return None
    df = df.copy()
    df["MCC_rank"] = df["MCC"].rank(ascending=False, method="min")
    df["F1_rank"] = df["F1"].rank(ascending=False, method="min")
    df["Combined_Rank"] = df["MCC_rank"] + df["F1_rank"]
    return df.sort_values(["Combined_Rank", "MCC"], ascending=[True, False]).reset_index(drop=True)


def validate_uploaded_data(data: pd.DataFrame, feature_cols: list, target_col: str, label_encoder):
    errors = []
    warnings = []

    if len(data) == 0:
        errors.append("The uploaded file has no rows.")
        return {"errors": errors, "warnings": warnings, "has_target": False}
    uploaded_cols = list(data.columns)
    uploaded_cols_stripped = {c.strip(): c for c in uploaded_cols}

    missing_cols = [c for c in feature_cols if c not in uploaded_cols]
    recoverable = [c for c in missing_cols if c in uploaded_cols_stripped]
    if recoverable:
        warnings.append(
            "Some column headers had extra whitespace and were auto-trimmed: "
            + ", ".join(recoverable)
        )
        data.rename(columns={uploaded_cols_stripped[c]: c for c in recoverable}, inplace=True)
        missing_cols = [c for c in missing_cols if c not in recoverable]

    if missing_cols:
        errors.append(
            "Missing required feature column: " + ", ".join(missing_cols)
        )

    extra_cols = [
        c for c in uploaded_cols
        if c not in feature_cols and c != target_col
    ]
    if extra_cols:
        warnings.append(
            "Uploaded file has extra column(s) not used by the model (ignored): "
            + ", ".join(extra_cols)
        )
    present_feature_cols = [c for c in feature_cols if c in data.columns]
    fully_null_cols = [c for c in present_feature_cols if data[c].isna().all()]
    if fully_null_cols:
        errors.append(
            "Feature column(s) are entirely empty/NaN: " + ", ".join(fully_null_cols)
        )

    partially_null_cols = [
        c for c in present_feature_cols
        if c not in fully_null_cols and data[c].isna().any()
    ]
    if partially_null_cols:
        warnings.append(
            "Feature column(s) contain some missing values, which may affect predictions: "
            + ", ".join(partially_null_cols)
        )

    has_target = target_col in data.columns
    if has_target:
        known_classes = set(label_encoder.classes_)
        target_values = set(data[target_col].dropna().astype(str))
        unknown_values = target_values - {str(c) for c in known_classes}
        if unknown_values:
            errors.append(
                f"Target column '{target_col}' contains value(s) not seen during "
                f"training: {sorted(unknown_values)}. Expected one of: {sorted(known_classes)}."
            )

    return {"errors": errors, "warnings": warnings, "has_target": has_target}


def get_positive_class_index(label_encoder, preferred_label="Yes"):
    classes = list(label_encoder.classes_)
    if preferred_label in classes:
        return classes.index(preferred_label)
    return len(classes) - 1


MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
}

meta = load_meta()
label_encoder = load_label_encoder()
summary_df = load_summary()
ranked_df = load_ranked_summary()

if meta is None or label_encoder is None:
    st.error(f"Missing model artifacts in `./{MODEL_DIR}` folder. Please run `train_models.py` first.")
    st.stop()

target_col = meta.get("target_col", "Churn")
feature_cols = meta.get("feature_cols", [])
positive_idx = get_positive_class_index(label_encoder, preferred_label="Yes")
positive_label = label_encoder.classes_[positive_idx]

# --------
# Sidebar
# --------
with st.sidebar:
    st.header("📊 Model Performance")

    uploaded_file = st.file_uploader(
        "Please Upload test data (CSV)",
        type=["csv"],
        help="Use test_data.csv or similar.",
    )

    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    model_choice = st.selectbox("Select a model", list(MODEL_FILES.keys()))

    st.divider()

    if ranked_df is not None and len(ranked_df) > 0:
        winner = ranked_df.iloc[0]
        st.markdown(
            f"""
            <div class="winner-card-light">
                <div class="winner-header-light">🏆 Best Model Winner (MCC + F1)</div>
                <hr style="border: 0; height: 1px; background: #cbd5e1; margin: 8px 0 12px 0;">
                <div class="winner-title-light">{winner['Model']}</div>
                <div class="winner-stats-light">
                    <b>MCC:</b> {winner['MCC']:.3f} &nbsp;|&nbsp; <b>F1:</b> {winner['F1']:.3f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.divider()

    with st.container(border=True):
        st.subheader("Rate My Work!")

        rating_choice = st.radio(
            "How you want to rate my Model Comparison results",
            options=["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐", "⭐"],
            index=None,
        )

        if rating_choice:
            st.success(f"Thank you for your feedback! You selected **{rating_choice}**")

# --------
# Main
# --------
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not parse the uploaded CSV: {e}")
        st.stop()

    with st.expander("📋 Expected file schema", expanded=False):
        st.write(f"**Required feature columns ({len(feature_cols)}):**")
        st.code(", ".join(feature_cols) or "(none defined in meta.json)")
        st.write(f"**Optional target column:** `{target_col}` (values: {list(label_encoder.classes_)})")

    validation = validate_uploaded_data(data, feature_cols, target_col, label_encoder)

    for w in validation["warnings"]:
        st.warning(f"⚠️ {w}")

    if validation["errors"]:
        for e in validation["errors"]:
            st.error(f"❌ {e}")
        st.stop()

    has_target = validation["has_target"]
    X = data[feature_cols]

    model_file = MODEL_FILES[model_choice]
    pipe = load_model(model_file)

    if pipe is None:
        st.error(f"Could not load pipeline file `{model_file}`.")
        st.stop()

    try:
        y_pred_encoded = pipe.predict(X)
        y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X)[:, positive_idx]
        elif hasattr(pipe, "decision_function"):
            y_proba = pipe.decision_function(X)
        else:
            st.error(f"Model `{model_choice}` exposes neither `predict_proba` nor `decision_function`.")
            st.stop()
    except Exception as e:
        st.error(
            f"Prediction failed for `{model_choice}`. This usually means the uploaded "
            f"columns don't match what the model was trained on.\n\nDetails: {e}"
        )
        st.stop()

    result_df = data.copy()
    result_df["Predicted_" + target_col] = y_pred_labels
    result_df["Churn_Probability"] = np.round(y_proba, 4)

    tab_overview, tab_predictions, tab_diagnostics, tab_compare = st.tabs(
        ["📊 Overview", "🔮 Predictions", "🩺 Model Diagnostics", "⚖️ Compare All Models"]
    )

    #  Overview tab 
    with tab_overview:
        st.subheader(f"Snapshot — {model_choice}")

        churn_count = int((y_pred_labels == positive_label).sum())
        churn_rate = churn_count / len(y_pred_labels) * 100 if len(y_pred_labels) else 0.0

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

    #  Predictions tab 
    with tab_predictions:
        st.subheader(f"Predictions — {model_choice}")

        styled = result_df.head(50).style.map(
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

    #  Diagnostics tab 
    with tab_diagnostics:
        if has_target:
            y_true_labels = data[target_col]
            try:
                y_true_encoded = label_encoder.transform(y_true_labels)
            except ValueError as e:
                st.error(
                    f"The '{target_col}' column contains values the model wasn't trained on "
                    f"(expected one of {list(label_encoder.classes_)}). Details: {e}"
                )
                st.stop()

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
                report_df.style.background_gradient(cmap="Blues"),
                use_container_width=True,
            )

            # Feature importance for tree-based models
            classifier = None
            if hasattr(pipe, "named_steps"):
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

    #  Compare All Models tab
    with tab_compare:
        st.subheader("All Models — Training-Time Comparison")
        if ranked_df is not None and len(ranked_df) > 0:
            st.caption(
                f"Official pick (MCC + F1 combined rank): **{ranked_df.iloc[0]['Model']}**. "
                "Use the metric picker below to see trade-offs on any single metric."
            )
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