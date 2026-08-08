import argparse
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
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

MODEL_DIR = "model"
RANDOM_STATE = 42
CV_FOLDS = 5


def nullCheck(df):
    totalNull = df.isnull().sum().sum()

    if totalNull > 0:
        print(f"Total Null values in given dataframe: {totalNull}")
        print("----------------------------")
        print("Percentage of missing values in Descending order (%)")

        # Calculate percentage by multiplying by 100
        missingVal = (df.isnull().sum() / len(df)) * 100
        missingVal = missingVal[missingVal > 0]
        missingVal.sort_values(ascending=False, inplace=True)

        return missingVal

    else:
        print("Given Data Frame has zero Null Values")
        return pd.Series(dtype=float)

def load_and_clean(path):
    df = pd.read_csv(path)
    # 1. Dataset Size
    print("--- Dataset Size ---")
    print("df data set has {0} rows and {1} columns".format(df.shape[0], df.shape[1]))
    print("------------------------------------------------")

    # Data cleaning
    # Total charges Coerced from string → numeric , then missing values filled with the median

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # customerID dropped entirely as this is a unique key with no importance in prediction
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df = df.dropna()
    return df


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


# Each entry is (estimator, param_grid). Grids are intentionally small -
# this is model comparison, not a full tuning pass, so we just nudge the
# handful of knobs that matter most for each algorithm. class_weight is
# set to "balanced" everywhere it's supported since churn is ~27% of
# the data and we don't want models defaulting to "always predict No".
def get_model_specs():
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"),
            {"classifier__C": [0.1, 1.0, 3.0]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            {"classifier__max_depth": [4, 6, 8, None], "classifier__min_samples_leaf": [1, 5, 10]},
        ),
        "kNN": (
            KNeighborsClassifier(),
            {"classifier__n_neighbors": [5, 7, 11, 15], "classifier__weights": ["uniform", "distance"]},
        ),
        "Naive Bayes": (
            GaussianNB(),
            {},  # nothing meaningful to grid-search here
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            {"classifier__n_estimators": [200, 400], "classifier__max_depth": [8, 12, None]},
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


def select_best_model(results_df):
    """MCC + F1 combined-rank winner. We don't use Accuracy/AUC here -
    with ~27% churn, a model that just predicts "No" for everyone still
    scores ~73% accuracy, so those two metrics can be misleading on
    their own."""
    df = results_df.copy()
    df["MCC_rank"] = df["MCC"].rank(ascending=False, method="min")
    df["F1_rank"] = df["F1"].rank(ascending=False, method="min")
    df["Combined_Rank"] = df["MCC_rank"] + df["F1_rank"]
    df["MCC_F1_Avg"] = (df["MCC"] + df["F1"]) / 2

    df = df.sort_values(["Combined_Rank", "MCC"], ascending=[True, False]).reset_index(drop=True)
    winner = df.iloc[0]["Model"]

    print("\n=== MCC + F1 Ranking ===")
    print(df[["Model", "MCC", "F1", "MCC_rank", "F1_rank", "Combined_Rank"]].round(4).to_string(index=False))
    print(f"\nWinner (best combined MCC + F1 rank): {winner}")
    return df


def main(args):
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Loading data from: {args.data}")
    df = load_and_clean(args.data)
    print(f"Checking NULL or MISSING Values in training DataSet \n")
    print(nullCheck(df))
    target_col = args.target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available columns: {list(df.columns)}")

    y_raw = df[target_col]
    X = df.drop(columns=[target_col])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )

    test_export = X_test.copy()
    test_export[target_col] = label_encoder.inverse_transform(y_test)
    test_export.to_csv("test_data.csv", index=False)
    print(f"Saved test_data.csv with shape {test_export.shape}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mcc_scorer = make_scorer(matthews_corrcoef)

    results = []
    cv_results = []
    specs = get_model_specs()

    for name, (clf, param_grid) in specs.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

        if param_grid:
            # small grid search, scored on MCC since that's what we care
            # about for the final ranking anyway
            search = GridSearchCV(pipe, param_grid, scoring=mcc_scorer, cv=cv, n_jobs=-1)
            search.fit(X_train, y_train)
            best_pipe = search.best_estimator_
            best_params = search.best_params_
            cv_mcc = search.best_score_
        else:
            # GaussianNB has nothing worth tuning, so just fit once and
            # get a CV score for comparability with the tuned models
            best_pipe = pipe.fit(X_train, y_train)
            best_params = {}
            cv_mcc = cross_val_score(pipe, X_train, y_train, scoring=mcc_scorer, cv=cv, n_jobs=-1).mean()

        cv_results.append({"Model": name, "CV_MCC": cv_mcc, "Best_Params": best_params})

        y_pred = best_pipe.predict(X_test)
        if hasattr(best_pipe, "predict_proba"):
            y_proba = best_pipe.predict_proba(X_test)[:, 1]
        else:
            y_proba = best_pipe.decision_function(X_test)

        metrics = evaluate(y_test, y_pred, y_proba)
        metrics["Model"] = name
        results.append(metrics)

        fname = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(best_pipe, os.path.join(MODEL_DIR, fname))
        print(f"Trained {name} (CV MCC={cv_mcc:.4f}, params={best_params}): {metrics}")

    results_df = pd.DataFrame(results)[["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    results_df.to_csv(os.path.join(MODEL_DIR, "metrics_summary.csv"), index=False)

    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(os.path.join(MODEL_DIR, "cv_summary.csv"), index=False)

    ranked_df = select_best_model(results_df)
    ranked_df.to_csv(os.path.join(MODEL_DIR, "ranked_summary.csv"), index=False)
    winner_model = ranked_df.iloc[0]["Model"]

    # No meta.json artifact is written. The Streamlit app derives target_col,
    # feature_cols, numeric_cols, and categorical_cols directly from a fitted
    # pipeline's ColumnTransformer at load time, and derives the winner_model
    # directly from ranked_summary.csv, so nothing here needs to be persisted
    # separately.

    print("\n=== Final Comparison Table (held-out test set) ===")
    print(results_df.round(4).to_string(index=False))
    print(f"\nOverall winner (by MCC + F1): {winner_model}")
    print("All models, metrics, and metadata saved to ./model/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data and df classification models.")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument("--target", type=str, default="Churn", help="Name of the target column (default: Churn)")
    args = parser.parse_args()
    main(args)
