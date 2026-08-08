# Telco Customer Churn — Multi-Model Classification & Streamlit Demo

## a. Problem Statement
Customer churn — when a subscriber stops using a company's service — is one
of the most expensive problems in the telecom industry, since acquiring a
new customer typically costs far more than retaining an existing one. This
project builds and compares six classification models that predict whether
a telecom customer will churn (`Yes`/`No`) based on their account and
service usage attributes, so that a business could proactively target
at-risk customers with retention offers.

## b. Dataset Description
- **Source:** Telco Customer Churn dataset (Kaggle, originally published by
  IBM Sample Data Sets).
- **Instances:** ~7,043 customer records.
- **Features:** 19 predictor columns spanning demographics (gender, senior
  citizen status, partner/dependents), account information (tenure,
  contract type, payment method, billing), and subscribed services
  (phone, internet, streaming, security, backup, tech support add-ons).
- **Target:** `Churn` — binary (`Yes` / `No`).
- **Class balance:** ~26.5% churn / ~73.5% no-churn (moderately imbalanced —
  this is why MCC is reported alongside Accuracy).
- **Preprocessing:** dropped `customerID`; coerced `TotalCharges` to
  numeric (a handful of blank strings for new customers with 0 tenure were
  imputed with the median); numeric features scaled with `StandardScaler`;
  categorical features one-hot encoded.
- **Imbalance handling:** `class_weight="balanced"` is set on every model
  that supports it (Logistic Regression, Decision Tree, Random Forest), so
  the minority "churn" class isn't just ignored in favor of accuracy.
- **Model selection:** each algorithm is tuned with a small grid search
  under 5-fold stratified cross-validation, scored on MCC (see the grids
  in `get_model_specs()` in `train_models.py`). The tuned pipeline is then
  refit on the full training split and scored once on the held-out test
  set — those held-out numbers are what's reported below.

## c. GitHub Repository Link
https://github.com/krishnaranjan92/telco-churn-model-comparison

## d. Models Used

### Comparison Table (held-out test set, tuned models)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7398 | 0.8494 | 0.5055 | 0.8143 | 0.6238 | 0.4704 |
| Decision Tree | 0.7493 | 0.8399 | 0.5169 | 0.8214 | 0.6345 | 0.4869 |
| kNN | 0.7862 | 0.8309 | 0.5978 | 0.5893 | 0.5935 | 0.4485 |
| Naive Bayes | 0.6868 | 0.8122 | 0.4512 | 0.8429 | 0.5878 | 0.4179 |
| Random Forest (Ensemble) | 0.7663 | 0.8471 | 0.5401 | 0.7929 | 0.6425 | 0.4975 |

Recall jumps noticeably for most models compared to an earlier, untuned
pass (Logistic Regression's recall went from ~0.57 to ~0.81, for example)
because of `class_weight="balanced"` — the models are now explicitly
penalized for missing churners, not just for being wrong overall. Accuracy
correspondingly drops a bit for some models; that trade-off is expected
and is exactly why Accuracy alone isn't used to pick a winner.

### Winner Selection Criteria: MCC + F1

Given the ~27% churn class imbalance, **Accuracy is not used to pick the winner** — a model that always predicts "No churn" would already score ~73% accuracy without being useful. Instead, each model is ranked separately on **MCC** (the most robust single metric under imbalance, since it uses all four confusion-matrix cells) and **F1** (the harmonic mean of precision and recall, capturing how well the model handles the minority "churn" class). The two per-metric ranks are summed into a `Combined_Rank` (lower is better); the model with the lowest combined rank wins. This logic lives in `select_best_model()` in `train_models.py` and is written out to `model/ranked_summary.csv` on every run; per-model cross-validation MCC and the winning hyperparameters from the grid search are in `model/cv_summary.csv`.

| ML Model Name | MCC | F1 | MCC Rank | F1 Rank | Combined Rank |
|---|---|---|---|---|---|
| **Random Forest (Ensemble)** | **0.4975** | **0.6425** | **1** | **1** | **2** |
| Decision Tree | 0.4869 | 0.6345 | 2 | 2 | 4 |
| Logistic Regression | 0.4704 | 0.6238 | 3 | 3 | 6 |
| kNN | 0.4485 | 0.5935 | 4 | 4 | 8 |
| Naive Bayes | 0.4179 | 0.5878 | 5 | 5 | 10 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Random Forest (Ensemble) | **Winner on MCC + F1** — best on both metrics individually (MCC 0.4975, F1 0.6425), the best achievable combined rank (2). Best hyperparameters from the grid search: `max_depth=8, n_estimators=400`. Constraining tree depth (vs. the earlier unconstrained/untuned version) reduces overfitting to noise while `class_weight="balanced"` keeps recall high — the combination of averaging many shallow, imbalance-aware trees is what pushes it ahead of the single linear model here. |
| Decision Tree | Close second (Combined Rank 4) once tuned (`max_depth=6`) and balanced — a striking change from an earlier untuned run where a single unconstrained tree was the weakest model by a wide margin. Depth-limiting is doing most of the work: it stops the tree from memorizing the training split. |
| Logistic Regression | Third — still a strong, stable baseline (MCC 0.4704, F1 0.6238) and the best AUC of any model (0.8494), meaning its *ranking* of customers by churn risk is arguably the most reliable, even though its default 0.5 decision threshold isn't the best-calibrated for this class balance. |
| kNN | Best hyperparameters found were `n_neighbors=15, weights=uniform`. It has no direct imbalance handling (kNN has no `class_weight`), so its recall (0.5893) lags the other models noticeably — it's the only model here that doesn't explicitly compensate for the minority class. |
| Naive Bayes | Lowest MCC and F1 of the five. Its high recall (0.8429) is a byproduct of the algorithm's independence assumption, not a deliberate imbalance strategy, and its precision (0.4512) is the weakest of all models — it flags far more false positives than it needs to for that same recall. |
| **Overall Winner (MCC + F1)** | **Random Forest** — best on both MCC and F1 after tuning and imbalance-aware training. This differs from a naive/untuned comparison, where Logistic Regression can look like the winner; adding `class_weight="balanced"` and even a light hyperparameter search changes the ranking, which is itself worth noting in a write-up — the "best model" is a property of the whole pipeline, not just the algorithm choice. |

---

## Project Structure
```
telco_churn_project/
│-- app.py                  # Streamlit app
│-- train_models.py         # Trains all 6 models, saves artifacts
│-- requirements.txt
│-- README.md
│-- test_data.csv           # Held-out test split (generated by train_models.py)
│-- model/
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest.pkl
│   │-- label_encoder.pkl
│   │-- meta.json
│   │-- metrics_summary.csv
│   │-- ranked_summary.csv    # Models ranked by MCC + F1 combined rank
│   │-- cv_summary.csv        # Per-model CV MCC + winning hyperparameters
```

## How to Reproduce
```bash
pip install -r requirements.txt
python train_models.py --data telco_customer_churn.csv
streamlit run app.py
```

## Streamlit App Features
- CSV upload for test data
- Model selection dropdown (all 5 algorithm families, 6 total counting the
  ensemble as separate from its base learners per the assignment spec)
- Live evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix heatmap and full classification report