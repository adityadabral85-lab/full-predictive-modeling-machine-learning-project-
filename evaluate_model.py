from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
REPORT_PATH = REPORTS_DIR / "model_report.md"


def evaluate_models(models, x_test, y_test):
    results = []

    for model_name, model in models.items():
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]

        results.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1_score": f1_score(y_test, predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probabilities),
            }
        )

    return pd.DataFrame(results).sort_values("roc_auc", ascending=False)


def save_confusion_matrix(model, x_test, y_test):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    predictions = model.predict(x_test)
    matrix = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(path)
    plt.close()
    return path


def save_roc_curve(model, x_test, y_test):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    probabilities = model.predict_proba(x_test)[:, 1]
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    auc_score = roc_auc_score(y_test, probabilities)

    plt.figure(figsize=(6, 5))
    plt.plot(false_positive_rate, true_positive_rate, label=f"ROC AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()

    path = FIGURES_DIR / "roc_curve.png"
    plt.savefig(path)
    plt.close()
    return path


def generate_report(results_df, best_model_name, model_path, figure_paths):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_rows = [
        "| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for _, row in results_df.iterrows():
        metrics_rows.append(
            (
                f"| {row['model']} | {row['accuracy']:.2f} | {row['precision']:.2f} | "
                f"{row['recall']:.2f} | {row['f1_score']:.2f} | {row['roc_auc']:.2f} |"
            )
        )

    metrics_table = "\n".join(metrics_rows)
    figures = "\n".join(f"- `{path.relative_to(PROJECT_ROOT)}`" for path in figure_paths)

    report = f"""# Predictive Modeling Report

## Goal

Predict customer churn using supervised machine learning models.

## Model Results

{metrics_table}

## Best Model

The best model is **{best_model_name}** based on ROC AUC score.

Saved model:

- `{model_path.relative_to(PROJECT_ROOT)}`

## Visualizations

{figures}

## Key Learning Outcome

This project demonstrates how to train, test, compare, and evaluate supervised
machine learning models using accuracy metrics, a confusion matrix, and a ROC curve.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    return REPORT_PATH
