"""Train the best-performing RandomForest on feature set A (baseline-only)
and produce SHAP plots for the test set.

Hyperparameters are taken from `02_modeling.ipynb` (commit 4774cae), which
reported Test AUROC = 0.6926 (95% CI 0.6229-0.7562) — the best AUC in the
project.
"""
from __future__ import annotations

import os
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

RNG = 42
DATA_DIR = "processed/A_baseline"
OUT_DIR = "results/shap_rf"
MODEL_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    X_tr = pd.read_parquet(f"{DATA_DIR}/X_train.parquet")
    X_val = pd.read_parquet(f"{DATA_DIR}/X_val.parquet")
    X_te = pd.read_parquet(f"{DATA_DIR}/X_test.parquet")
    y_tr = np.load(f"{DATA_DIR}/y_train.npy")
    y_val = np.load(f"{DATA_DIR}/y_val.npy")
    y_te = np.load(f"{DATA_DIR}/y_test.npy")
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def train_rf(X_tr, y_tr):
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RNG,
    )
    rf.fit(X_tr, y_tr)
    return rf


def evaluate(rf, X_val, y_val, X_te, y_te):
    p_val = rf.predict_proba(X_val)[:, 1]
    p_te = rf.predict_proba(X_te)[:, 1]
    return {
        "val_auroc": float(roc_auc_score(y_val, p_val)),
        "val_auprc": float(average_precision_score(y_val, p_val)),
        "test_auroc": float(roc_auc_score(y_te, p_te)),
        "test_auprc": float(average_precision_score(y_te, p_te)),
    }


def compute_shap(rf, X_te):
    """TreeExplainer returns shap values for the positive class."""
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(X_te, check_additivity=False)
    if isinstance(sv, list):
        shap_pos = sv[1]
        base_pos = explainer.expected_value[1]
    else:
        arr = np.asarray(sv)
        if arr.ndim == 3:
            shap_pos = arr[..., 1]
            base_pos = explainer.expected_value
            if isinstance(base_pos, (list, np.ndarray)):
                base_pos = np.asarray(base_pos).ravel()[1]
        else:
            shap_pos = arr
            base_pos = explainer.expected_value
            if isinstance(base_pos, (list, np.ndarray)):
                base_pos = np.asarray(base_pos).ravel()[0]
    return shap_pos, float(base_pos)


def plot_bar(shap_pos, X_te, path, top_n=20):
    mean_abs = np.abs(shap_pos).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]
    feats = X_te.columns[order]
    vals = mean_abs[order]

    plt.figure(figsize=(8, 0.35 * top_n + 1))
    plt.barh(range(top_n), vals[::-1], color="#4C72B0")
    plt.yticks(range(top_n), feats[::-1])
    plt.xlabel("mean(|SHAP value|)  —  global feature importance")
    plt.title("RandomForest (A_baseline) — SHAP top-20 features")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    return pd.DataFrame({"feature": feats, "mean_abs_shap": vals})


def plot_beeswarm(shap_pos, X_te, path, top_n=20):
    plt.figure()
    shap.summary_plot(
        shap_pos,
        X_te,
        max_display=top_n,
        show=False,
        plot_size=(9, 0.35 * top_n + 1),
    )
    plt.title("RandomForest (A_baseline) — SHAP beeswarm (top-20)")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_dependence(shap_pos, X_te, top_features, path):
    n = len(top_features)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    axes = np.atleast_2d(axes).ravel()
    for i, feat in enumerate(top_features):
        shap.dependence_plot(
            feat,
            shap_pos,
            X_te,
            ax=axes[i],
            show=False,
            interaction_index="auto",
        )
        axes[i].set_title(feat, fontsize=10)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(
        "RandomForest (A_baseline) — SHAP dependence (top-6 features)", y=1.02
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_data()
    print(f"train={X_tr.shape}  val={X_val.shape}  test={X_te.shape}")
    print(f"pos rates: train={y_tr.mean():.3f}  val={y_val.mean():.3f}  test={y_te.mean():.3f}")

    rf = train_rf(X_tr, y_tr)
    metrics = evaluate(rf, X_val, y_val, X_te, y_te)
    print("Metrics:", json.dumps(metrics, indent=2))

    joblib.dump(rf, f"{MODEL_DIR}/rf_A_baseline.joblib")

    print("Computing SHAP values on test set...")
    shap_pos, base = compute_shap(rf, X_te)
    print(f"SHAP shape: {shap_pos.shape}  base value: {base:.4f}")

    bar_path = f"{OUT_DIR}/shap_top20_bar.png"
    bee_path = f"{OUT_DIR}/shap_top20_beeswarm.png"
    dep_path = f"{OUT_DIR}/shap_dependence_top6.png"

    top_table = plot_bar(shap_pos, X_te, bar_path, top_n=20)
    top_table.to_csv(f"{OUT_DIR}/shap_top20_features.csv", index=False)
    plot_beeswarm(shap_pos, X_te, bee_path, top_n=20)

    top6 = top_table["feature"].head(6).tolist()
    plot_dependence(shap_pos, X_te, top6, dep_path)

    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump({**metrics, "shap_base_value": base}, f, indent=2)

    np.save(f"{OUT_DIR}/shap_values_test.npy", shap_pos)

    print(f"\nSaved:\n  {bar_path}\n  {bee_path}\n  {dep_path}\n  {OUT_DIR}/shap_top20_features.csv")
    print("\nTop-10 features by mean(|SHAP|):")
    print(top_table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
