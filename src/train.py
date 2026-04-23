"""
Model Training
==============
Trains three classifiers on the engineered features, compares them, and
saves the best model + encoders to models/.

Classifiers compared
--------------------
  1. Logistic Regression  — fast baseline, interpretable coefficients
  2. Random Forest        — handles non-linearity, no scaling needed
  3. XGBoost              — gradient boosting, typically best on tabular data

Target
------
  is_delayed  (1 = delay > 5 min, 0 = on time / early)

Evaluation metrics
------------------
  Accuracy, Precision, Recall, F1 (macro), ROC-AUC
  Confusion matrix saved as PNG for each model.
  Feature-importance chart saved for RF and XGBoost.

Outputs
-------
  models/best_model.joblib     — fitted best classifier
  models/encoders.joblib       — fitted feature encoders (from features.py)
  models/results_summary.csv   — metrics table for all three models
  data/fig_confusion_*.png     — confusion matrix PNGs
  data/fig_feature_importance.png

Usage
-----
  python src/train.py                  # default: use full dataset
  python src/train.py --sample 50000  # faster run on a subset
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
from matplotlib.patches import Rectangle

# Local import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_features, save_encoders

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT / "data" / "training_data.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

DB_RED  = "#CC0000"
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "axes.titleweight": "bold"})

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _build_models() -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=4,   # ~(115298/26189) class imbalance weight
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate(
    name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
) -> dict:
    """Run predictions, compute metrics, save confusion matrix PNG."""
    y_pred  = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    metrics = {
        "Model":     name,
        "Accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "Precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "Recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "F1":        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "ROC-AUC":   round(float(roc_auc_score(y_test, y_proba)), 4) if y_proba is not None else None,
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Reds", ax=ax,
        xticklabels=["On time", "Delayed"],
        yticklabels=["On time", "Delayed"],
        linewidths=0.5,
    )
    ax.set_title(f"Confusion Matrix — {name}", pad=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    slug = name.lower().replace(" ", "_")
    fig.savefig(ROOT / "data" / f"fig_confusion_{slug}.png", bbox_inches="tight")
    plt.close(fig)

    # Classification report
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:<12}: {v}")
    print(classification_report(y_test, y_pred, target_names=["On time", "Delayed"]))

    return metrics


def _plot_feature_importance(
    models: dict,
    feature_names: list[str],
) -> None:
    """Side-by-side feature importance for RF and XGBoost."""
    tree_models = {
        k: v for k, v in models.items()
        if hasattr(v, "feature_importances_")
    }
    if not tree_models:
        return

    n = len(tree_models)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, tree_models.items()):
        importances = model.feature_importances_
        idx = np.argsort(importances)
        ax.barh(
            [feature_names[i] for i in idx],
            importances[idx],
            color=DB_RED,
        )
        ax.set_title(f"Feature Importance — {name}")
        ax.set_xlabel("Importance")

    plt.suptitle("Feature Importances", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(ROOT / "data" / "fig_feature_importance.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nFeature importance chart saved.")


def _plot_all_metrics(results: list[dict]) -> None:
    """Bar chart comparing all models on all metrics."""
    res_df = pd.DataFrame(results).set_index("Model")
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    res_df = res_df[metrics_to_plot]

    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(metrics_to_plot))
    width = 0.25
    colors = ["#4878CF", "#6ACC65", DB_RED]

    for i, (model_name, row) in enumerate(res_df.iterrows()):
        ax.bar(x + i * width, np.array(row.values, dtype=float), width, label=model_name, color=colors[i])

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics", pad=12)
    ax.legend()
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)

    # Annotate bars — isinstance narrows type from Patch → Rectangle
    for bar in ax.patches:
        if not isinstance(bar, Rectangle):
            continue
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.005,
                f"{h:.3f}",
                ha="center", va="bottom", fontsize=7,
            )

    plt.tight_layout()
    fig.savefig(ROOT / "data" / "fig_model_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("Model comparison chart saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(sample: int | None) -> None:
    # ── Load data ────────────────────────────────────────────────────────────
    print(f"\nLoading data from {DATA_PATH} …")
    df = pd.read_csv(DATA_PATH)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42)
        print(f"  Sampled {len(df):,} rows.")
    else:
        print(f"  {len(df):,} rows loaded.")

    # ── Feature engineering ──────────────────────────────────────────────────
    print("\nEngineering features …")
    X, y, encoders = build_features(df)
    feature_names = list(X.columns)
    print(f"  X shape: {X.shape}  |  class balance: {y.value_counts().to_dict()}")

    # ── Train / test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train = pd.DataFrame(X_train)
    X_test  = pd.DataFrame(X_test)
    y_train = pd.Series(y_train)
    y_test  = pd.Series(y_test)
    print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")

    # ── Train all models ─────────────────────────────────────────────────────
    models     = _build_models()
    results    = []
    fitted     = {}

    for name, model in models.items():
        print(f"\n[{name}] Training …")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        print(f"  Done in {elapsed:.1f}s")
        fitted[name] = model
        metrics = _evaluate(name, model, X_test, y_test, feature_names)
        metrics["train_time_s"] = round(elapsed, 1)
        results.append(metrics)

    # ── Pick best model by F1 ────────────────────────────────────────────────
    best_row   = max(results, key=lambda r: r["F1"])
    best_name  = best_row["Model"]
    best_model = fitted[best_name]
    print(f"\n{'='*60}")
    print(f"  Best model: {best_name}  (F1 = {best_row['F1']})")
    print(f"{'='*60}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    model_path   = MODELS_DIR / "best_model.joblib"
    encoder_path = MODELS_DIR / "encoders.joblib"
    results_path = MODELS_DIR / "results_summary.csv"

    joblib.dump(best_model, model_path)
    save_encoders(encoders, encoder_path)

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)

    print(f"\nSaved:")
    print(f"  {model_path}")
    print(f"  {encoder_path}")
    print(f"  {results_path}")

    # ── Charts ───────────────────────────────────────────────────────────────
    _plot_feature_importance(fitted, feature_names)
    _plot_all_metrics(results)

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    print(results_df[display_cols].to_string(index=False))
    print("="*60)

    # Store best model name in a small metadata file so the API can read it
    meta_path = MODELS_DIR / "best_model_meta.joblib"
    joblib.dump({"name": best_name, "metrics": best_row, "features": feature_names}, meta_path)
    print(f"\nMetadata saved → {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DB delay prediction models.")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional: train on a random subset of N rows (faster for testing)",
    )
    args = parser.parse_args()
    main(sample=args.sample)
