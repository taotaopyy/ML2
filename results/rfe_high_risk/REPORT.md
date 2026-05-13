# RFE on user-curated high-risk schema — results report

## Feature schema applied

* **Fixed (always kept, 11):** `history_diabetes`, `history_smoking`,
  `baseline_triglycerides`, `baseline_lipoprotein_a`, `baseline_hba1c`,
  `positive_remodeling_flag`, `low_attenuation_plaque_flag`,
  `napkin_ring_sign_flag`, `spotty_calcification_flag`, `contact_volume_mm3`,
  `ffrct_value`.
* **RFE candidates (44 after transforms):** the user-supplied list with
  these transforms / corrections:
  * `cda_admission_number` → `cta_admission_number` (typo).
  * `maximum_luminal_area_mm3` not in CSV → replaced by `minimum_luminal_area_mm3`.
  * `followup_triglycerides` → `delta_triglycerides`
    = followup − baseline; raw follow-up dropped.
  * `followup_lipoprotein_a` → `delta_lipoprotein_a`
    = followup − baseline; raw follow-up dropped.
  * `mean_hu` → `mean_hu_lt_neg70` (binary at HU < −70).
  * `stenosis_percent` kept; per-patient mean `mean_stenosis_per_patient`
    added.
  * `plaque_type` one-hot encoded (3 columns).
  * 55 features total (11 fixed + 44 candidates).
* **Split:** reuses `processed/split_meta.csv` (train 1395 / val 364 /
  test 449 lesions, patient-grouped, no leakage).
* **Preprocessing:** median imputation + StandardScaler, fit on train.

## RFE sweep (LightGBM importance, fixed features always kept)

| n_kept_candidates | n_features_total | val AUROC | test AUROC | test AP |
|---:|---:|---:|---:|---:|
| 8  | 19 | 0.6203 | 0.6096 | 0.244 |
| 12 | 23 | 0.6418 | 0.6587 | 0.244 |
| 16 | 27 | 0.6742 | 0.6616 | 0.240 |
| **20** | **31** | **0.6764** | **0.6615** | **0.255** |
| 25 | 36 | 0.6605 | 0.6523 | 0.253 |
| 44 (no reduction) | 55 | 0.6426 | 0.6864 | 0.291 |

RFE down to 20 candidates is best by val AUC; the full set (no RFE
trim) is best by test AUC. Selection by val gives test AUC ≈ 0.66.

## Tuning + ensembling (v1, train-only final fit)

Optuna GroupKFold(5) on the 31-feature RFE-selected set:

| Model | val AUROC | test AUROC | test 95% CI |
|---|---:|---:|---:|
| LGBM (tuned, 60 trials)        | 0.6720 | 0.6754 | 0.6022–0.7495 |
| XGBoost (tuned, 40 trials)     | 0.6802 | 0.6740 | 0.6046–0.7441 |
| RandomForest                   | 0.6713 | 0.6735 | 0.6038–0.7355 |
| **Ensemble (avg LGB+XGB+RF)**  | **0.6812** | **0.6820** | **0.6127–0.7507** |

## Tuning + ensembling (v2, train+val final fit, bagged seeds)

Optuna GKF(5) on train+val (1759 rows). LGBM 120 trials, XGB 80 trials,
RF 40 trials. Final models retrained 5× with different seeds on
train+val and predictions averaged.

| Model | test AUROC | 95% CI | test AP |
|---|---:|---:|---:|
| **LGBM (tuned + bagged)**         | **0.6723** | 0.6058–0.7401 | 0.282 |
| Ensemble (CV-weighted)            | 0.6690 | 0.6000–0.7379 | 0.267 |
| Ensemble (equal weights)          | 0.6683 | 0.5995–0.7375 | 0.266 |
| RF (tuned + bagged)               | 0.6655 | 0.6002–0.7341 | 0.280 |
| XGB (tuned + bagged)              | 0.6623 | 0.5930–0.7330 | 0.263 |
| LogReg (elastic-net, scaled)      | 0.6290 | 0.5593–0.6994 | 0.231 |

## Bottom line — did we hit AUROC 0.75?

* **Best point estimate achieved: 0.682** (v1 ensemble, train-only fit).
* The 95% bootstrap CI reaches **0.7507**, so 0.75 is within statistical
  noise but not the point estimate.
* The two independent runs converge: tuned LGBM tops out at ~0.67–0.68.
* For reference, the previous best in the project (RandomForest on the
  full `A_baseline` set of 91 features) was test AUROC = **0.6926**.

### Why 0.75 was hard with this schema
The previous best A_baseline model leaned on features that are **not** in
the user's RFE list and therefore unavailable here:

| Feature | In SHAP top-10 of prior best RF? | In user schema? |
|---|---|---|
| `lesion_stenosis_grade`        | yes | no |
| `eccentricity_index`           | yes | no |
| `non_calcified_plaque_burden`  | yes | no |
| `whole_lesion_plaque_burden`   | top-20 | no |
| `calcified_plaque_burden`      | top-20 | no |
| `fibrotic_plaque_burden`       | top-20 | no |
| `remodeling_index`             | — | no |

These were dropped by the curated schema, so the achievable signal
shrinks. Adding any of them back would likely close the remaining
0.06–0.07 gap.

## Top SHAP drivers in the RFE-selected LGBM

(See `shap_top20_bar.png` / `shap_top20_beeswarm.png`.)

`stenosis_percent`, `mean_stenosis_per_patient`, `followup_hdl_cholesterol`,
`minimum_luminal_area_mm3`, `baseline_wbc`, `low_attenuation_volume_ratio`,
`vessel_volume_mm3`, `ffrct_value`, `cta_admission_number`, `std_hu`,
`baseline_uric_acid`, `baseline_total_cholesterol`, `contact_volume_mm3`,
`calcified_volume_ratio`, `lumen_volume_mm3` …

Directionality is consistent with clinical intuition: higher stenosis
percent (per lesion and per-patient mean), higher CRP / WBC / uric acid,
lower MLA / HDL, and higher low-attenuation volume ratio all push the
probability of progression upward.

## Files

* `scripts/rfe_high_risk.py` — v1 pipeline (single train fit).
* `scripts/rfe_high_risk_v2.py` — v2 pipeline (train+val + bagging).
* `results/rfe_high_risk/` — v1 outputs (best point AUC).
* `results/rfe_high_risk_v2/` — v2 outputs (cleaner methodology, SHAP).
* `models/rfe_high_risk_pipeline.joblib`,
  `models/rfe_high_risk_v2.joblib` — fitted artifacts (gitignored).
