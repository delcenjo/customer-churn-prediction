import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split

from .config import FIGURES_DIR, MODEL_PATH, RANDOM_STATE, TARGET, TEST_SIZE
from .data import load_dataset


def load_test_split():
    df = load_dataset()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return X_test, y_test


def plot_confusion_matrix(model, X_test, y_test):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def plot_roc_curve(model, X_test, y_test):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_title("ROC curve")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=120)
    plt.close(fig)


def plot_feature_importance(model, X_test, y_test, top_n=15):
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE,
        scoring="roc_auc", n_jobs=-1,
    )
    names = np.array(X_test.columns)
    order = np.argsort(result.importances_mean)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(range(len(order)), result.importances_mean[order][::-1])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names[order][::-1], fontsize=9)
    ax.set_xlabel("Permutation importance (ROC-AUC drop)")
    ax.set_title(f"Top {len(order)} features")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model = joblib.load(MODEL_PATH)
    X_test, y_test = load_test_split()
    plot_confusion_matrix(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)
    plot_feature_importance(model, X_test, y_test)
    print(f"Saved figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
