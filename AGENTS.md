# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

This is a medical ML research project for coronary artery plaque progression prediction using CTA + clinical data. The codebase consists of a Jupyter notebook (`01_preprocessing.ipynb`) that preprocesses a CSV dataset and outputs train/val/test splits in `.parquet`/`.npy` format under `processed/`.

### Running the notebook

```bash
# Execute non-interactively (headless):
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=300 01_preprocessing.ipynb --output /tmp/executed.ipynb

# Or start Jupyter Lab for interactive use:
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token="" --NotebookApp.password="" --allow-root
```

### Dependencies

All required packages (`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`, `jupyter`) are pre-installed in the VM environment. There is no `requirements.txt` in the repo.

### Linting / testing

There is no formal linting or test framework configured. To validate the notebook runs correctly, use the `jupyter nbconvert --execute` command above — a zero exit code confirms successful execution.

### Key notes

- The dataset CSV (`final_paired_analysis_ml.csv`) is committed to the repo and must be present for the notebook to run.
- Outputs go to `processed/` with subdirs `A_baseline/`, `B_baseline_delta/`, `C_all/` (each containing `.parquet` features, `.npy` labels, and a `.joblib` preprocessor).
- The notebook uses `GroupShuffleSplit` by `accession_id` to prevent data leakage between patients.
