# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Medical/clinical ML research project for coronary artery plaque progression prediction. Single Jupyter notebook (`01_preprocessing.ipynb`) handles data preprocessing; downstream modeling uses the processed data in `processed/` (feature sets A/B/C with train/val/test splits).

### Running the application

- **Dev server**: `jupyter lab --no-browser --ip=0.0.0.0 --port=8888 --ServerApp.token="" --ServerApp.password=""` from `/workspace`
- **Execute notebook**: `jupyter nbconvert --to notebook --execute 01_preprocessing.ipynb --output 01_preprocessing_executed.ipynb --ExecutePreprocessor.timeout=300`

### Lint

- `nbqa ruff 01_preprocessing.ipynb` — lint the notebook with ruff via nbqa

### Testing

No automated test suite exists. Verify correctness by executing the notebook end-to-end (see above) and confirming the processed output files are generated in `processed/`.

### Key notes

- No `requirements.txt` or `pyproject.toml` exists; dependencies are implicit from notebook imports: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`, `pyarrow`.
- The dataset (`final_paired_analysis_ml.csv`) and processed outputs (`processed/`) are committed to the repo.
- Python 3.12 is used. All packages are pip-installed.
- The `01_preprocessing_executed.ipynb` file is a transient artifact from notebook execution and should not be committed.
