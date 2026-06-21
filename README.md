# Customer Churn Prediction

![CI](https://github.com/delcenjo/customer-churn-prediction/actions/workflows/ci.yml/badge.svg)
[![Kaggle notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/delcenjo/telco-customer-churn-ml)

Keeping an existing telecom customer is much cheaper than winning a new one, so
the useful question for a retention team is not "who left last month" but "who is
about to leave". This project builds a model that scores each customer by their
probability of churning, so the people running retention campaigns can spend
their budget on the accounts most at risk instead of contacting everyone.

Churn is a minority event here (roughly 26% of customers), and that shapes the
whole evaluation. Plain accuracy is misleading on imbalanced data: a model that
predicts "nobody churns" would already be right about three quarters of the time
and be completely useless. So the models are compared on ROC-AUC, which measures
how well they rank a churner above a non-churner regardless of where you set the
decision threshold.

## The data

The Telco Customer Churn dataset: 7,043 customers described by 21 columns
covering demographics, the services they subscribe to, their contract type and
their billing. The raw CSV is not committed to the repo. Download it with
`python scripts/download_data.py` (details in [data/README.md](data/README.md)).

A couple of quirks worth knowing about before modelling:

- `TotalCharges` comes through as text and is blank for brand-new customers
  (tenure 0). It gets coerced to a number, with those blanks filled as 0.0.
- `customerID` is just an identifier and is dropped so the model cannot memorise it.

## How it works

The cleaning logic lives in `churn.data`. From there the flow is:

1. Numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`)
   are standardised and everything else is one-hot encoded. This happens inside a
   `ColumnTransformer` (`churn.features`) that is bundled into the same
   scikit-learn `Pipeline` as the model, so the scaler and encoder only ever see
   the training fold and no information leaks from the test data.
2. Three classifiers are put up against each other with 5-fold cross-validated
   ROC-AUC: logistic regression as the baseline, a random forest, and histogram
   gradient boosting. The first two run with balanced class weights to counter
   the imbalance.
3. The best of the three is refit on the full training split and scored once on a
   held-out test set. Figures for the ROC curve, confusion matrix and permutation
   importance are saved alongside a `metrics.json`.

If you would rather read through the analysis than run it, the same EDA and model
comparison is written up as a runnable
[Kaggle notebook](https://www.kaggle.com/code/delcenjo/telco-customer-churn-ml).

## What the numbers say

Cross-validated ROC-AUC on the training set:

| Model               | CV ROC-AUC      |
| ------------------- | --------------- |
| Logistic regression | 0.845 ± 0.014   |
| Gradient boosting   | 0.833 ± 0.011   |
| Random forest       | 0.825 ± 0.013   |

The simple logistic regression comes out on top, which is a good reminder that
the fancier model is not always the better one. It is retrained on the full
training split and evaluated on the held-out test set (1,409 customers):

| Metric    | Score |
| --------- | ----- |
| ROC-AUC   | 0.842 |
| Accuracy  | 0.738 |
| Precision | 0.504 |
| Recall    | 0.783 |
| F1        | 0.614 |

Because of the balanced class weights, the model leans towards recall: it catches
about 78% of the customers who really do churn, at the cost of more false alarms.
For retention that is usually the trade you want, since missing a customer who
walks away is far more expensive than an unnecessary "please stay" offer to one
who was not going anywhere.

| Exploratory analysis | Model evaluation |
| --- | --- |
| ![Churn rate by contract](reports/figures/churn_by_contract.png) | ![ROC curve](reports/figures/roc_curve.png) |
| ![Tenure by churn](reports/figures/tenure_by_churn.png) | ![Feature importance](reports/figures/feature_importance.png) |

The figures point at the same story: churn concentrates among customers on
month-to-month contracts, with short tenure and high monthly charges. People on
one- and two-year contracts churn far less, which makes contract length the most
actionable lever a retention team actually has. Full numbers are in
[reports/metrics.json](reports/metrics.json).

## Running it yourself

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python scripts/download_data.py     # fetch the dataset
python -m churn.eda                 # exploratory figures
python -m churn.train               # compare models, train and persist the best
python -m churn.evaluate            # evaluation figures
pytest                              # run the tests
```

## Where things live

```
src/churn/
  config.py      paths and constants
  data.py        loading and cleaning
  features.py    preprocessing pipeline
  eda.py         exploratory figures
  train.py       model comparison, training, persistence
  evaluate.py    evaluation figures
tests/           unit tests for data and features
scripts/         dataset download
reports/         metrics and figures
```

If this were taken further, the obvious next steps would be tuning the decision
threshold around the real cost of a missed churner and adding probability
calibration.
