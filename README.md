# Telco Customer Churn — Multi-Model Classification & Streamlit web application to demonstrate models


## a. Problem Statement
As part of this dataset "Telecom Customer Churn" we predict ahead of time whether customers are going to churn so we can flag at risk customers early and do something about them like sending a retention offer, or a call from support to understand the issue and concern. This project trains and compares six classification models to find the best model that can predict, given a telecom customer's account details and the services they've signed up for, whether they're going to churn.

## b. Dataset Description
- **Source:** Kaggle
- **Source URL:** `path = kagglehub.dataset_download("blastchar/telco-customer-churn")`
- **Size:** ~7,043 customer records.
- **Features:** 19 columns (customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges).
- **Target:** `Churn` — binary (`Yes` / `No`).
- **Class balance:** ~26.5% churn / ~73.5% no-churn (moderately imbalanced).
- **Preprocessing:**
- **Feature Removal**: Dropped customerID as it is a non-predictive unique identifier.
- **Data Cleaning:** Coerced **TotalCharges** to numeric, replaced blank strings (present in zero-tenure records) with the **median** value.
- **Numerical Scaling:** Standardized continuous numerical features using StandardScaler to ensure zero mean and unit variance.
- **Categorical Encoding:** Converted all categorical features into binary indicator variables using One-Hot Encoding.
- **Imbalance handling:** `class_weight="balanced"` to supports  (Logistic Regression, Decision Tree, Random Forest), so
  the minority "churn" class isn't just ignored in favor of accuracy.
- **Model selection:** As this is an imbalanced classification problem, the best model is picked using MCC and F1 Score.

## c. GitHub Repository Link
- URL:
https://github.com/krishnaranjan92/telco-churn-model-comparison
- List of Merged PRs:
https://github.com/krishnaranjan92/telco-churn-model-comparison/pulls?q=is%3AMerged+is%3Apr+

## d. Models Used

### Comparison Table (held-out test set, tuned models)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7398 | 0.8494 | 0.5055 | 0.8143 | 0.6238 | 0.4704 |
| Decision Tree | 0.7493 | 0.8399 | 0.5169 | 0.8214 | 0.6345 | 0.4869 |
| kNN | 0.7862 | 0.8309 | 0.5978 | 0.5893 | 0.5935 | 0.4485 |
| Naive Bayes | 0.6868 | 0.8122 | 0.4512 | 0.8429 | 0.5878 | 0.4179 |
| Random Forest (Ensemble) | 0.7663 | 0.8471 | 0.5401 | 0.7929 | 0.6425 | 0.4975 |



### Winner Selection Criteria: MCC + F1

Given the ~27% churn class imbalance, **Accuracy is not used to pick the winner** — so we are selecting the winner by **MCC + F1 score**. Each model is ranked separately on MCC and on F1; the two ranks are added together into a Combined Rank (lower is better), and the model with the lowest Combined Rank wins.

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
| Naive Bayes | Lowest MCC and F1 of the five. Its high recall (0.8429) is a byproduct of the algorithm's independence assumption, not a deliberate imbalance strategy, and its precision (0.4512) is the weakest of all models — it flags far more false positives than it needs to for that same recall. **Caveat:** GaussianNB assumes features are conditionally independent given the class, but that assumption is violated here — one-hot-encoded columns from the same underlying attribute are mechanically correlated (e.g. `InternetService_No` implies `OnlineSecurity_No`, `OnlineBackup_No`, `TechSupport_No`, `StreamingTV_No`, and `StreamingMovies_No` are all also 1, since those services don't exist without internet). Treating correlated features as independent causes the model to double-count the same evidence, which likely contributes to its weak precision here, on top of it being the only model with no imbalance handling. |
| **Overall Winner (MCC + F1)** | **Random Forest** — best on both MCC and F1 after tuning and imbalance-aware training. This differs from a naive/untuned comparison, where Logistic Regression can look like the winner; adding `class_weight="balanced"` and even a light hyperparameter search changes the ranking, which is itself worth noting in a write-up — the "best model" is a property of the whole pipeline, not just the algorithm choice. |

---

## Project Structure
```
telco_churn_project/
│-- app.py                  # Streamlit app
│-- train_models.py         # Trains all 6 models, saves artifacts
│-- requirements.txt
│-- README.md
│-- Telco-Customer-Churn.csv #Actual Dataset on which models are trained.
│-- test_data.csv           # Held-out test split (generated by train_models.py)
│-- model/
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest.pkl
│   │-- label_encoder.pkl
│   │-- metrics_summary.csv
│   │-- ranked_summary.csv    # Models ranked by MCC + F1 combined rank
│   │-- cv_summary.csv        # Per-model CV MCC + winning hyperparameters
```

## Requirement.txt 
- streamlit>=1.38
- scikit-learn>=1.5
- numpy>=1.26
- pandas>=2.2
- joblib>=1.4
- plotly>=5.22

## How to Run
```bash
pip install -r requirements.txt
python train_models.py --data Telco-Customer-Churn.csv
streamlit run app.py
```

## Streamlit App Features
- CSV upload for test data
- Model selection dropdown (all 5 algorithm families, 6 total counting the
  ensemble as separate from its base learners per the assignment spec)
- Live evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix heatmap and full classification report
