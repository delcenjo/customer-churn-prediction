import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from .config import FIGURES_DIR, TARGET
from .data import load_dataset


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = load_dataset()
    labelled = df.copy()
    labelled["Churn"] = labelled[TARGET].map({0: "No", 1: "Yes"})

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=labelled, x="Churn", hue="Churn", palette="Set2", legend=False, ax=ax)
    ax.set_title("Target distribution (churn)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "target_distribution.png", dpi=120)
    plt.close(fig)

    rate = df.groupby("Contract")[TARGET].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=rate.values, y=rate.index, hue=rate.index, palette="Set2", legend=False, ax=ax)
    ax.set_xlabel("Churn rate")
    ax.set_title("Churn rate by contract type")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "churn_by_contract.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=labelled, x="tenure", hue="Churn", bins=30, palette="Set2", element="step", ax=ax)
    ax.set_title("Tenure distribution by churn")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "tenure_by_churn.png", dpi=120)
    plt.close(fig)

    print(f"Saved EDA figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
