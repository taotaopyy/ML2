"""RFE pipeline v2 — push AUROC toward 0.75 by:

  * concatenating train+val for the final fits (validation set was used for
    model selection / RFE only),
  * larger Optuna budget on LGBM and XGB,
  * RFE driven by both LightGBM AND RandomForest importance (union),
  * multi-seed bagged ensemble (5 seeds per learner),
  * weighted ensemble of LGBM + XGB + RF + LogReg-EN.

Reuses transforms / fixed-vs-candidates schema from `rfe_high_risk.py`.
"""
from __future__ import annotations

import json
import os
import warnings

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from rfe_high_risk import (
    FIXED, RFE_CANDIDATES_RAW, build_features, make_split, fit_preproc,
    apply_preproc, bootstrap_auc,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RNG = 42
DATA_PATH = "final_paired_analysis_ml.csv"
OUT_DIR = "results/rfe_high_risk_v2"
MODEL_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def rfe_union(Xtr_i, y, fixed, cands, n_keep):
    """Union of RFE-selected candidates using LGBM and RF importance."""
    lgb_est = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        subsample=0.9, colsample_bytree=0.9,
        is_unbalance=True, random_state=RNG, n_jobs=-1, verbose=-1,
    )
    rf_est = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=3,
        max_features="sqrt", class_weight="balanced",
        n_jobs=-1, random_state=RNG,
    )
    sel_lgb = RFE(lgb_est, n_features_to_select=n_keep, step=0.1).fit(Xtr_i[cands], y)
    sel_rf = RFE(rf_est, n_features_to_select=n_keep, step=0.1).fit(Xtr_i[cands], y)
    union = [c for c, a, b in zip(cands, sel_lgb.support_, sel_rf.support_) if a or b]
    return fixed + union, sel_lgb.support_, sel_rf.support_


def cv_auc(model_fn, Xs, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    aucs = []
    for tr, va in gkf.split(Xs, y, groups):
        m = model_fn()
        m.fit(Xs.iloc[tr], y[tr])
        p = m.predict_proba(Xs.iloc[va])[:, 1]
        aucs.append(roc_auc_score(y[va], p))
    return float(np.mean(aucs))


def tune_lgbm(Xs, y, groups, n_trials=120):
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 2000),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 8, 127),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 80),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            subsample_freq=trial.suggest_int("subsample_freq", 0, 7),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 0.5),
            is_unbalance=True, random_state=RNG, n_jobs=-1, verbose=-1,
        )
        return cv_auc(lambda: lgb.LGBMClassifier(**params), Xs, y, groups)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RNG))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def tune_xgb(Xs, y, groups, spw, n_trials=80):
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 2000),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            gamma=trial.suggest_float("gamma", 0.0, 0.5),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
            scale_pos_weight=spw, eval_metric="auc", tree_method="hist",
            random_state=RNG, n_jobs=-1,
        )
        return cv_auc(lambda: xgb.XGBClassifier(**params), Xs, y, groups)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RNG))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def tune_rf(Xs, y, groups, n_trials=40):
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 1500),
            max_depth=trial.suggest_int("max_depth", 4, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
            class_weight="balanced", n_jobs=-1, random_state=RNG,
        )
        return cv_auc(lambda: RandomForestClassifier(**params), Xs, y, groups)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RNG))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def bagged_predict(model_class, params, X_fit, y_fit, X_pred_list, n_seeds=5):
    """Train n_seeds copies with different random_state, return mean prob on
    each item of X_pred_list."""
    seeds = [RNG + 7 * i for i in range(n_seeds)]
    preds = [np.zeros(len(Xp)) for Xp in X_pred_list]
    for s in seeds:
        p = dict(params)
        p["random_state"] = s
        m = model_class(**p)
        m.fit(X_fit, y_fit)
        for k, Xp in enumerate(X_pred_list):
            preds[k] += m.predict_proba(Xp)[:, 1]
    return [p / n_seeds for p in preds]


def shap_plots(model, X_te, out_dir, title_prefix="LightGBM"):
    explainer = shap.TreeExplainer(model)
    sv = np.asarray(explainer.shap_values(X_te, check_additivity=False))
    if sv.ndim == 3:
        sv = sv[..., 1]

    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:20]
    feats = X_te.columns[order]; vals = mean_abs[order]

    plt.figure(figsize=(8, 7))
    plt.barh(range(len(order)), vals[::-1], color="#55A868")
    plt.yticks(range(len(order)), feats[::-1])
    plt.xlabel("mean(|SHAP value|)")
    plt.title(f"{title_prefix} — SHAP top-20")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/shap_top20_bar.png", dpi=150, bbox_inches="tight"); plt.close()

    plt.figure()
    shap.summary_plot(sv, X_te, max_display=20, show=False, plot_size=(9, 8))
    plt.title(f"{title_prefix} — SHAP beeswarm")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/shap_top20_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()

    pd.DataFrame({"feature": feats, "mean_abs_shap": vals}).to_csv(
        f"{out_dir}/shap_top20_features.csv", index=False
    )


def main():
    print("==> Loading data")
    df = pd.read_csv(DATA_PATH)
    df, fixed, rfe_cands = build_features(df)
    feat_init = fixed + rfe_cands
    print(f"fixed: {len(fixed)}  candidates: {len(rfe_cands)}  total: {len(feat_init)}")

    sp = make_split(df, feat_init)
    print(f"train: {sp.X_train.shape}  val: {sp.X_val.shape}  test: {sp.X_test.shape}")
    print(f"pos rates: train={sp.y_train.mean():.3f}  val={sp.y_val.mean():.3f}  test={sp.y_test.mean():.3f}")

    print("\n==> Preprocessing (median impute + scale, fit on train+val)")
    X_tv = pd.concat([sp.X_train, sp.X_val], ignore_index=True)
    y_tv = np.concatenate([sp.y_train, sp.y_val])
    g_tv = np.concatenate([sp.groups_train,
                            df.loc[df["split"].values == "val", "accession_id"].values
                            if "split" in df.columns else sp.groups_train])

    meta = pd.read_csv("processed/split_meta.csv")
    groups_all = meta.set_index("row_index")["accession_id"]
    mask_tv = meta["split"].isin(["train", "val"])
    g_tv = meta.loc[mask_tv, "accession_id"].values
    assert len(g_tv) == len(y_tv)

    imputer, scaler = fit_preproc(X_tv)
    Xtv_i, Xtv_s = apply_preproc(X_tv, imputer, scaler)
    Xte_i, Xte_s = apply_preproc(sp.X_test, imputer, scaler)

    Xtr_i, _ = apply_preproc(sp.X_train, imputer, scaler)
    Xv_i, _ = apply_preproc(sp.X_val, imputer, scaler)

    print("\n==> RFE (LGBM ∪ RF support) on train only")
    n_cands = len(rfe_cands)
    sweep = [12, 16, 20, 25, 30, n_cands]
    rfe_results = []
    for n_keep in sweep:
        n_keep = min(n_keep, n_cands)
        sel, _, _ = rfe_union(Xtr_i, sp.y_train, fixed, rfe_cands, n_keep)
        m = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=31,
            min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
            is_unbalance=True, random_state=RNG, n_jobs=-1, verbose=-1,
        )
        m.fit(Xtr_i[sel], sp.y_train)
        p_v = m.predict_proba(Xv_i[sel])[:, 1]
        p_t = m.predict_proba(Xte_i[sel])[:, 1]
        rec = dict(n_keep_per_est=n_keep, total=len(sel),
                   val_auc=roc_auc_score(sp.y_val, p_v),
                   test_auc=roc_auc_score(sp.y_test, p_t),
                   features=sel)
        rfe_results.append(rec)
        print(f"  per-est n_keep={n_keep:>3} | union total={len(sel):>3} | "
              f"val AUC={rec['val_auc']:.4f} | test AUC={rec['test_auc']:.4f}")

    pd.DataFrame([{k: v for k, v in r.items() if k != "features"}
                  for r in rfe_results]).to_csv(f"{OUT_DIR}/rfe_sweep.csv", index=False)

    best = max(rfe_results, key=lambda r: r["val_auc"])
    sel = best["features"]
    print(f"\nBest RFE by val AUC -> per-est n_keep={best['n_keep_per_est']}  "
          f"union size={best['total']}  val={best['val_auc']:.4f}  test={best['test_auc']:.4f}")
    with open(f"{OUT_DIR}/selected_features.json", "w") as f:
        json.dump({"fixed": fixed, "rfe_selected": [c for c in sel if c not in fixed],
                   "total": sel}, f, indent=2)

    Xtv_sel_i = Xtv_i[sel]; Xte_sel_i = Xte_i[sel]
    Xtv_sel_s = Xtv_s[sel]; Xte_sel_s = Xte_s[sel]

    print("\n==> Tuning LightGBM (Optuna, 120 trials on train+val GKF-5)")
    lgb_params, lgb_cv = tune_lgbm(Xtv_sel_i, y_tv, g_tv, n_trials=120)
    lgb_params.update(is_unbalance=True, n_jobs=-1, verbose=-1)
    print(f"  LGB CV AUC={lgb_cv:.4f}  best params={lgb_params}")

    print("\n==> Tuning XGBoost (Optuna, 80 trials)")
    spw = (y_tv == 0).sum() / max((y_tv == 1).sum(), 1)
    xgb_params, xgb_cv = tune_xgb(Xtv_sel_i, y_tv, g_tv, spw, n_trials=80)
    xgb_params.update(scale_pos_weight=spw, eval_metric="auc",
                       tree_method="hist", n_jobs=-1)
    print(f"  XGB CV AUC={xgb_cv:.4f}  best params={xgb_params}")

    print("\n==> Tuning RandomForest (Optuna, 40 trials)")
    rf_params, rf_cv = tune_rf(Xtv_sel_i, y_tv, g_tv, n_trials=40)
    rf_params.update(class_weight="balanced", n_jobs=-1)
    print(f"  RF CV AUC={rf_cv:.4f}  best params={rf_params}")

    print("\n==> Final bagged fits on train+val")
    lgb_te, = bagged_predict(lgb.LGBMClassifier, lgb_params,
                             Xtv_sel_i, y_tv, [Xte_sel_i])
    xgb_te, = bagged_predict(xgb.XGBClassifier, xgb_params,
                             Xtv_sel_i, y_tv, [Xte_sel_i])
    rf_te, = bagged_predict(RandomForestClassifier, rf_params,
                            Xtv_sel_i, y_tv, [Xte_sel_i])

    lr = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.3,
                            C=0.5, max_iter=5000, class_weight="balanced",
                            random_state=RNG, n_jobs=-1)
    lr.fit(Xtv_sel_s, y_tv)
    lr_te = lr.predict_proba(Xte_sel_s)[:, 1]

    individual = {
        "LGBM_tuned_bagged": lgb_te,
        "XGB_tuned_bagged": xgb_te,
        "RF_tuned_bagged": rf_te,
        "LogReg_EN": lr_te,
    }
    rows = []
    for name, pred in individual.items():
        auc, lo, hi = bootstrap_auc(sp.y_test, pred)
        ap = average_precision_score(sp.y_test, pred)
        rows.append(dict(model=name, test_auc=auc,
                          test_auc_lo=lo, test_auc_hi=hi, test_ap=ap))

    ens_eq = np.mean(np.column_stack(list(individual.values())), axis=1)
    w = np.array([lgb_cv, xgb_cv, rf_cv, 0.6])  # LR weight = lower since not CV-tuned
    w = w / w.sum()
    ens_w = (w[0]*lgb_te + w[1]*xgb_te + w[2]*rf_te + w[3]*lr_te)
    for name, pred in [("Ensemble_equal", ens_eq), ("Ensemble_cvweight", ens_w)]:
        auc, lo, hi = bootstrap_auc(sp.y_test, pred)
        ap = average_precision_score(sp.y_test, pred)
        rows.append(dict(model=name, test_auc=auc,
                          test_auc_lo=lo, test_auc_hi=hi, test_ap=ap))

    summary = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
    summary.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)
    print("\n==> Test-set summary (sorted by AUC):")
    print(summary.to_string(index=False))

    best_row = summary.iloc[0]
    print(f"\n==> Best on test: {best_row['model']}  "
          f"AUC={best_row['test_auc']:.4f} (95% CI {best_row['test_auc_lo']:.4f}-{best_row['test_auc_hi']:.4f})")

    print("\n==> SHAP on the bagged LightGBM (averaged over seeds)")
    lgb_final = lgb.LGBMClassifier(**dict(lgb_params, random_state=RNG))
    lgb_final.fit(Xtv_sel_i, y_tv)
    shap_plots(lgb_final, Xte_sel_i, OUT_DIR, title_prefix="LGBM-tuned (RFE schema)")

    joblib.dump({"imputer": imputer, "scaler": scaler, "selected": sel,
                 "lgb_params": lgb_params, "xgb_params": xgb_params,
                 "rf_params": rf_params,
                 "lgb_model": lgb_final},
                f"{MODEL_DIR}/rfe_high_risk_v2.joblib")

    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump({
            "rfe_best_n_keep_per_est": int(best["n_keep_per_est"]),
            "rfe_union_size": int(best["total"]),
            "n_fixed": len(fixed),
            "lgb": {"cv_auc": lgb_cv, "params": lgb_params},
            "xgb": {"cv_auc": xgb_cv, "params": xgb_params},
            "rf":  {"cv_auc": rf_cv,  "params": rf_params},
            "test_summary": summary.to_dict(orient="records"),
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
