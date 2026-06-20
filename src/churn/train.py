import json

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from .config import (
    CV_FOLDS,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from .data import load_dataset
from .features import build_preprocessor


def candidate_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def make_pipeline(model, X):
    return Pipeline([("preprocessor", build_preprocessor(X)), ("classifier", model)])


def compute_test_metrics(pipeline, X_test, y_test):
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    cv_results = {}
    best_name, best_score = None, -1.0
    for name, model in candidate_models().items():
        pipeline = make_pipeline(model, X_train)
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=CV_FOLDS, scoring="roc_auc", n_jobs=-1
        )
        cv_results[name] = {
            "cv_roc_auc_mean": float(scores.mean()),
            "cv_roc_auc_std": float(scores.std()),
        }
        print(f"{name:>22}  ROC-AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")
        if scores.mean() > best_score:
            best_name, best_score = name, scores.mean()

    best_pipeline = make_pipeline(candidate_models()[best_name], X_train)
    best_pipeline.fit(X_train, y_train)
    metrics = compute_test_metrics(best_pipeline, X_test, y_test)

    joblib.dump(best_pipeline, MODEL_PATH)
    report = {
        "best_model": best_name,
        "cross_validation": cv_results,
        "test_metrics": metrics,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2))

    print(f"\nBest model: {best_name}")
    for name, value in metrics.items():
        print(f"  {name:>10} = {value:.4f}")
    print(f"\nSaved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
