# Customer Churn Prediction

End-to-end machine learning pipeline that predicts which telecom customers are
likely to churn, so retention efforts can be focused where they matter most.

The project covers the full workflow: data cleaning, exploratory analysis,
a reusable preprocessing pipeline, model comparison with cross-validation, and
evaluation on a held-out test set.

## Problem

Customer acquisition is far more expensive than retention. Given a customer's
contract, services and billing profile, the goal is to estimate the probability
of churn and identify the factors that drive it. The target is imbalanced
(~26% churn), so models are compared with **ROC-AUC** rather than accuracy.

## Dataset

Telco Customer Churn — 7,043 customers and 21 columns (demographics, subscribed
services, contract and billing). The raw file is not versioned; download it with
`python scripts/download_data.py` (see [data/README.md](data/README.md)).

## Approach

1. **Cleaning** — drop the identifier, coerce `TotalCharges` to numeric (blank for
   new customers), encode the target.
2. **Preprocessing** — a `ColumnTransformer` scales numeric features and
   one-hot-encodes categoricals, wrapped in a `Pipeline` to prevent data leakage.
3. **Modelling** — logistic regression (baseline), random forest and gradient
   boosting, compared with 5-fold cross-validated ROC-AUC.
4. **Evaluation** — the best model is refit and scored on the test set; figures
   for the ROC curve, confusion matrix and permutation importance are generated.

## Project structure

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

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python scripts/download_data.py     # fetch the dataset
python -m churn.eda                 # exploratory figures
python -m churn.train               # compare models, train and persist the best
python -m churn.evaluate            # evaluation figures
pytest                              # run the tests
```

## Results

_Populated by `python -m churn.train`; see [reports/metrics.json](reports/metrics.json)._

## Possible improvements

- Threshold tuning driven by the business cost of false negatives.
- Probability calibration and SHAP-based explanations.
- Packaging the model behind a small inference API.
