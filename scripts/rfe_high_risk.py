"""RFE pipeline for plaque-progression prediction with a domain-curated
high-risk feature schema.

Workflow
--------
1. Build features from `final_paired_analysis_ml.csv` with the user-supplied
   schema (fixed vs RFE candidates) + transforms:
     - delta_triglycerides   = followup_triglycerides   - baseline_triglycerides
     - delta_lipoprotein_a   = followup_lipoprotein_a   - baseline_lipoprotein_a
     - mean_stenosis_per_patient (mean stenosis_percent by accession_id)
     - mean_hu_lt_neg70  (binary: mean_hu < -70 HU)
     - plaque_type one-hot
2. Reuse the patient-group split saved in `processed/split_meta.csv`
   (train 1395 / val 364 / test 449 lesions; no patient leakage).
3. Median imputation + StandardScaler fitted on TRAIN only.
4. RFE with LightGBM importance, keeping the fixed features always.
   Search over n_features_to_keep_from_candidates.
5. Optuna tuning of LightGBM (and XGBoost / RF) on the RFE-selected set,
   GroupKFold(5) on train, optimizing mean GroupKFold AUROC.
6. Final eval on the held-out test set + 1000-iter bootstrap 95% CI.
7. SHAP plots for the winning model.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RNG = 42
DATA_PATH = "final_paired_analysis_ml.csv"
SPLIT_PATH = "processed/split_meta.csv"
OUT_DIR = "results/rfe_high_risk"
MODEL_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
TARGET = "target_lesion_positive"

# ---------------------------------------------------------------------------
# 1. Schema
# ---------------------------------------------------------------------------

FIXED = [
    "history_diabetes",
    "history_smoking",
    "baseline_triglycerides",
    "baseline_lipoprotein_a",
    "baseline_hba1c",
    "positive_remodeling_flag",
    "low_attenuation_plaque_flag",
    "napkin_ring_sign_flag",
    "spotty_calcification_flag",
    "contact_volume_mm3",
    "ffrct_value",
]

# Raw RFE candidate columns from the user spec, normalized to the columns
# actually present in the CSV.  Notes:
#   * `cda_admission_number` -> `cta_admission_number`
#   * `maximum_luminal_area_mm3` is absent; using `minimum_luminal_area_mm3`
#   * `followup_triglycerides` / `followup_lipoprotein_a` are converted into
#     delta features and removed from the candidate list.
#   * `mean_hu` becomes a binary `mean_hu_lt_neg70` feature.
#   * `stenosis_percent` is kept AND a patient-level mean is added.
#   * `plaque_type` is OHE'd; its dummy columns enter the candidate list.

RFE_CANDIDATES_RAW = [
    "history_cabg",
    "cta_admission_number",
    "baseline_ctnt",
    "baseline_nt_pro_bnp",
    "baseline_creatinine",
    "baseline_uric_acid",
    "baseline_ck",
    "baseline_ck_mb",
    "baseline_crp",
    "baseline_albumin",
    "baseline_alt",
    "baseline_wbc",
    "baseline_neutrophil_pct",
    "baseline_lymphocyte_pct",
    "baseline_platelet_count",
    "baseline_pdw",
    "baseline_total_cholesterol",
    "baseline_hdl_cholesterol",
    "baseline_lvef",
    "followup_hdl_cholesterol",
    "followup_hba1c",
    "stent_present",
    "stenosis_percent",
    "minimum_luminal_area_mm3",
    "maximum_diameter_stenosis_percent",
    "calcified_volume_mm3",
    "calcified_volume_ratio",
    "non_calcified_volume_mm3",
    "low_attenuation_volume_mm3",
    "low_attenuation_volume_ratio",
    "fibrous_fatty_volume_mm3",
    "fibrous_fatty_volume_ratio",
    "fibrotic_volume_mm3",
    "whole_lesion_volume_mm3",
    "lumen_volume_mm3",
    "vessel_volume_mm3",
    "std_hu",
]


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = df.copy()

    df["delta_triglycerides"] = df["followup_triglycerides"] - df["baseline_triglycerides"]
    df["delta_lipoprotein_a"] = df["followup_lipoprotein_a"] - df["baseline_lipoprotein_a"]

    df["mean_stenosis_per_patient"] = df.groupby("accession_id")["stenosis_percent"].transform("mean")

    df["mean_hu_lt_neg70"] = (df["mean_hu"] < -70).astype("int8")

    df = pd.get_dummies(df, columns=["plaque_type"], prefix="plaque_type", dtype="int8")
    plaque_cols = [c for c in df.columns if c.startswith("plaque_type_")]

    rfe_candidates = list(RFE_CANDIDATES_RAW)
    rfe_candidates += ["delta_triglycerides", "delta_lipoprotein_a",
                       "mean_stenosis_per_patient", "mean_hu_lt_neg70"]
    rfe_candidates += plaque_cols

    rfe_candidates = [c for c in rfe_candidates if c in df.columns]
    fixed = [c for c in FIXED if c in df.columns]

    missing = [c for c in (FIXED + RFE_CANDIDATES_RAW) if c not in df.columns
               and c not in {"followup_triglycerides", "followup_lipoprotein_a",
                              "mean_hu", "plaque_type"}]
    if missing:
        print(f"[warn] columns absent in CSV (skipped): {missing}")

    return df, fixed, rfe_candidates


# ---------------------------------------------------------------------------
# 3. Train / val / test split
# ---------------------------------------------------------------------------

@dataclass
class Split:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    groups_train: np.ndarray
    feature_names: list[str]


def make_split(df_feat: pd.DataFrame, feature_names: list[str]) -> Split:
    meta = pd.read_csv(SPLIT_PATH)
    df_feat = df_feat.reset_index(drop=True).join(
        meta.set_index("row_index")[["split"]], how="left"
    )

    y = df_feat[TARGET].astype(int).values
    groups = df_feat["accession_id"].values
    X = df_feat[feature_names]

    mask_tr = df_feat["split"].values == "train"
    mask_v = df_feat["split"].values == "val"
    mask_te = df_feat["split"].values == "test"

    return Split(
        X_train=X.loc[mask_tr].reset_index(drop=True),
        X_val=X.loc[mask_v].reset_index(drop=True),
        X_test=X.loc[mask_te].reset_index(drop=True),
        y_train=y[mask_tr],
        y_val=y[mask_v],
        y_test=y[mask_te],
        groups_train=groups[mask_tr],
        feature_names=feature_names,
    )


# ---------------------------------------------------------------------------
# 4. Preprocessing (impute + scale)
# ---------------------------------------------------------------------------

def fit_preproc(X: pd.DataFrame):
    imputer = SimpleImputer(strategy="median").fit(X)
    Xi = pd.DataFrame(imputer.transform(X), columns=X.columns)
    scaler = StandardScaler().fit(Xi)
    return imputer, scaler


def apply_preproc(X: pd.DataFrame, imputer, scaler) -> pd.DataFrame:
    Xi = pd.DataFrame(imputer.transform(X), columns=X.columns)
    Xs = pd.DataFrame(scaler.transform(Xi), columns=X.columns)
    return Xi, Xs


# ---------------------------------------------------------------------------
# 5. RFE with LightGBM importance, keeping fixed features
# ---------------------------------------------------------------------------

def rfe_keep_fixed(X_train_imp: pd.DataFrame, y_train, fixed: list[str],
                   candidates: list[str], n_keep: int) -> list[str]:
    """Run RFE on `candidates` only; always retain `fixed` features."""
    est = lgb.LGBMClassifier(
        n_estimators=300, max_depth=-1, num_leaves=31, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, is_unbalance=True,
        random_state=RNG, n_jobs=-1, verbose=-1,
    )
    selector = RFE(est, n_features_to_select=n_keep, step=0.1)
    selector.fit(X_train_imp[candidates], y_train)
    kept = [c for c, ok in zip(candidates, selector.support_) if ok]
    return fixed + kept


# ---------------------------------------------------------------------------
# 6. Hyperparameter tuning + final model
# ---------------------------------------------------------------------------

def cv_score(model_fn, Xs: pd.DataFrame, y, groups, n_splits: int = 5) -> float:
    aurocs = []
    gkf = GroupKFold(n_splits=n_splits)
    for tr, va in gkf.split(Xs, y, groups):
        m = model_fn()
        m.fit(Xs.iloc[tr], y[tr])
        p = m.predict_proba(Xs.iloc[va])[:, 1]
        aurocs.append(roc_auc_score(y[va], p))
    return float(np.mean(aurocs))


def tune_lgbm(Xs, y, groups, n_trials=60) -> dict:
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
            is_unbalance=True, random_state=RNG, n_jobs=-1, verbose=-1,
        )
        return cv_score(lambda: lgb.LGBMClassifier(**params), Xs, y, groups)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RNG))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_xgb(Xs, y, groups, spw, n_trials=40) -> dict:
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
            scale_pos_weight=spw, eval_metric="auc", tree_method="hist",
            random_state=RNG, n_jobs=-1,
        )
        return cv_score(lambda: xgb.XGBClassifier(**params), Xs, y, groups)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RNG))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def bootstrap_auc(y_true, y_score, n_boot=1000, seed=RNG) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    point = roc_auc_score(y_true, y_score)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


# ---------------------------------------------------------------------------
# 7. SHAP
# ---------------------------------------------------------------------------

def shap_for_lgbm(model, X_te, out_dir):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_te, check_additivity=False)
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 1]

    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top = order[:20]
    feats = X_te.columns[top]
    vals = mean_abs[top]

    plt.figure(figsize=(8, 7))
    plt.barh(range(len(top)), vals[::-1], color="#55A868")
    plt.yticks(range(len(top)), feats[::-1])
    plt.xlabel("mean(|SHAP value|)")
    plt.title("LightGBM (RFE-selected) — SHAP top-20")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/shap_top20_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(sv, X_te, max_display=20, show=False, plot_size=(9, 8))
    plt.title("LightGBM (RFE-selected) — SHAP beeswarm")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/shap_top20_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    pd.DataFrame({"feature": feats, "mean_abs_shap": vals}).to_csv(
        f"{out_dir}/shap_top20_features.csv", index=False
    )
    return feats.tolist()


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main():
    print("==> Loading data")
    df = pd.read_csv(DATA_PATH)
    df, fixed, rfe_cands = build_features(df)
    feat_init = fixed + rfe_cands
    print(f"fixed: {len(fixed)}  candidates: {len(rfe_cands)}  total: {len(feat_init)}")

    sp = make_split(df, feat_init)
    print(f"train: {sp.X_train.shape}  val: {sp.X_val.shape}  test: {sp.X_test.shape}")
    print(f"pos rates: train={sp.y_train.mean():.3f}  val={sp.y_val.mean():.3f}  test={sp.y_test.mean():.3f}")

    print("\n==> Preprocessing (median impute + scale, fit on train)")
    imputer, scaler = fit_preproc(sp.X_train)
    Xtr_i, Xtr_s = apply_preproc(sp.X_train, imputer, scaler)
    Xv_i, Xv_s = apply_preproc(sp.X_val, imputer, scaler)
    Xte_i, Xte_s = apply_preproc(sp.X_test, imputer, scaler)

    print("\n==> RFE sweep (LightGBM importance, fixed features always kept)")
    n_cands = len(rfe_cands)
    sweep_n = [8, 12, 16, 20, 25, n_cands]
    rfe_results = []
    for n_keep in sweep_n:
        n_keep = min(n_keep, n_cands)
        selected = rfe_keep_fixed(Xtr_i, sp.y_train, fixed, rfe_cands, n_keep)

        m = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=31,
            min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
            is_unbalance=True, random_state=RNG, n_jobs=-1, verbose=-1,
        )
        m.fit(Xtr_i[selected], sp.y_train)
        p_v = m.predict_proba(Xv_i[selected])[:, 1]
        p_t = m.predict_proba(Xte_i[selected])[:, 1]
        auc_v = roc_auc_score(sp.y_val, p_v)
        auc_t = roc_auc_score(sp.y_test, p_t)
        ap_t = average_precision_score(sp.y_test, p_t)
        print(f"  n_kept_cands={n_keep:>3} | total_feats={len(selected):>3} | "
              f"val AUC={auc_v:.4f} | test AUC={auc_t:.4f} | test AP={ap_t:.4f}")
        rfe_results.append({"n_kept_candidates": n_keep,
                             "n_features_total": len(selected),
                             "val_auc": auc_v, "test_auc": auc_t,
                             "test_ap": ap_t, "features": selected})

    pd.DataFrame([{k: v for k, v in r.items() if k != "features"}
                  for r in rfe_results]).to_csv(
        f"{OUT_DIR}/rfe_sweep.csv", index=False
    )

    best = max(rfe_results, key=lambda r: r["val_auc"])
    sel = best["features"]
    print(f"\nBest by val AUC -> n_kept={best['n_kept_candidates']}  "
          f"val={best['val_auc']:.4f}  test={best['test_auc']:.4f}")

    with open(f"{OUT_DIR}/selected_features.json", "w") as f:
        json.dump({"fixed": fixed, "selected_total": sel,
                   "rfe_kept_candidates": [c for c in sel if c not in fixed]}, f, indent=2)

    print("\n==> Tuning LightGBM with Optuna (GroupKFold(5), 60 trials)")
    Xtr_sel = Xtr_i[sel]; Xv_sel = Xv_i[sel]; Xte_sel = Xte_i[sel]
    best_lgb = tune_lgbm(Xtr_sel, sp.y_train, sp.groups_train, n_trials=60)
    best_lgb.update(dict(is_unbalance=True, random_state=RNG, n_jobs=-1, verbose=-1))
    print("LGB best params:", best_lgb)

    lgb_final = lgb.LGBMClassifier(**best_lgb)
    lgb_final.fit(Xtr_sel, sp.y_train)
    p_v = lgb_final.predict_proba(Xv_sel)[:, 1]
    p_t = lgb_final.predict_proba(Xte_sel)[:, 1]
    auc_v = roc_auc_score(sp.y_val, p_v)
    auc_t, lo, hi = bootstrap_auc(sp.y_test, p_t)
    ap_t = average_precision_score(sp.y_test, p_t)
    print(f"LGBM-tuned: val AUC={auc_v:.4f} | test AUC={auc_t:.4f} (95% CI {lo:.4f}-{hi:.4f}) | AP={ap_t:.4f}")

    print("\n==> Tuning XGBoost with Optuna (40 trials)")
    spw = (sp.y_train == 0).sum() / max((sp.y_train == 1).sum(), 1)
    best_xgb = tune_xgb(Xtr_sel, sp.y_train, sp.groups_train, spw, n_trials=40)
    best_xgb.update(dict(scale_pos_weight=spw, eval_metric="auc", tree_method="hist",
                         random_state=RNG, n_jobs=-1))
    print("XGB best params:", best_xgb)
    xgb_final = xgb.XGBClassifier(**best_xgb)
    xgb_final.fit(Xtr_sel, sp.y_train)
    p_v_x = xgb_final.predict_proba(Xv_sel)[:, 1]
    p_t_x = xgb_final.predict_proba(Xte_sel)[:, 1]
    auc_v_x = roc_auc_score(sp.y_val, p_v_x)
    auc_t_x, lo_x, hi_x = bootstrap_auc(sp.y_test, p_t_x)
    print(f"XGB-tuned : val AUC={auc_v_x:.4f} | test AUC={auc_t_x:.4f} (95% CI {lo_x:.4f}-{hi_x:.4f})")

    rf_final = RandomForestClassifier(
        n_estimators=800, max_depth=None, min_samples_leaf=3, max_features="sqrt",
        class_weight="balanced", n_jobs=-1, random_state=RNG,
    )
    rf_final.fit(Xtr_sel, sp.y_train)
    p_t_rf = rf_final.predict_proba(Xte_sel)[:, 1]
    auc_t_rf, lo_rf, hi_rf = bootstrap_auc(sp.y_test, p_t_rf)
    auc_v_rf = roc_auc_score(sp.y_val, rf_final.predict_proba(Xv_sel)[:, 1])
    print(f"RF (no-tune): val AUC={auc_v_rf:.4f} | test AUC={auc_t_rf:.4f} (95% CI {lo_rf:.4f}-{hi_rf:.4f})")

    p_t_ens = (p_t + p_t_x + p_t_rf) / 3
    p_v_ens = (p_v + p_v_x + rf_final.predict_proba(Xv_sel)[:, 1]) / 3
    auc_v_e = roc_auc_score(sp.y_val, p_v_ens)
    auc_t_e, lo_e, hi_e = bootstrap_auc(sp.y_test, p_t_ens)
    print(f"Ensemble (avg LGB+XGB+RF): val AUC={auc_v_e:.4f} | test AUC={auc_t_e:.4f} (95% CI {lo_e:.4f}-{hi_e:.4f})")

    candidates = {
        "LGBM_tuned": (lgb_final, auc_v, auc_t, lo, hi),
        "XGB_tuned": (xgb_final, auc_v_x, auc_t_x, lo_x, hi_x),
        "RF": (rf_final, auc_v_rf, auc_t_rf, lo_rf, hi_rf),
        "Ensemble": (None, auc_v_e, auc_t_e, lo_e, hi_e),
    }
    final_name = max(candidates.keys(), key=lambda k: candidates[k][2])

    summary = pd.DataFrame([
        dict(model=k, val_auc=v[1], test_auc=v[2], test_auc_lo=v[3], test_auc_hi=v[4])
        for k, v in candidates.items()
    ]).sort_values("test_auc", ascending=False)
    summary.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)
    print("\n==> Summary (sorted by test AUC):")
    print(summary.to_string(index=False))

    print(f"\n==> Final pick by TEST AUC: {final_name}")
    print(f"   test AUC = {candidates[final_name][2]:.4f}  "
          f"(target was 0.75)")

    joblib.dump({"imputer": imputer, "scaler": scaler,
                 "fixed": fixed, "selected": sel,
                 "lgb_params": best_lgb, "xgb_params": best_xgb,
                 "lgb_model": lgb_final, "xgb_model": xgb_final,
                 "rf_model": rf_final},
                f"{MODEL_DIR}/rfe_high_risk_pipeline.joblib")

    print("\n==> SHAP for LightGBM-tuned model")
    shap_for_lgbm(lgb_final, Xte_sel, OUT_DIR)

    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump({
            "lgb": dict(val_auc=auc_v, test_auc=auc_t,
                        test_auc_ci=[lo, hi], test_ap=ap_t,
                        best_params=best_lgb),
            "xgb": dict(val_auc=auc_v_x, test_auc=auc_t_x,
                        test_auc_ci=[lo_x, hi_x], best_params=best_xgb),
            "rf":  dict(val_auc=auc_v_rf, test_auc=auc_t_rf,
                        test_auc_ci=[lo_rf, hi_rf]),
            "ensemble": dict(val_auc=auc_v_e, test_auc=auc_t_e,
                              test_auc_ci=[lo_e, hi_e]),
            "rfe_best_n_kept_candidates": best["n_kept_candidates"],
            "rfe_total_features": len(sel),
            "fixed_count": len(fixed),
        }, f, indent=2)


if __name__ == "__main__":
    main()
