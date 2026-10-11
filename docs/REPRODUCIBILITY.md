# Reproducibility and data

- Notebook source: `notebooks/spaceship_titanic_experiments.ipynb` (47 cells, original notebook and outputs retained).
- Kaggle files: download `train.csv` and `test.csv` from [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/data). They are not included in this update. Place both alongside the notebook or adjust the paths.
- The notebook imports pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost, LightGBM, Optuna, MLflow and Jupyter. Package versions need to match the user's runtime; no independent dependency lockfile exists for this notebook.
- `mlflow.set_tracking_uri("file:./mlruns")` writes experiment data locally. MLflow screenshots capture a particular past run state, not a bundled tracking database.
- The original notebook's train/test preprocessing is **not** a production-ready fitted transformer; avoid blindly reusing model artifacts with independently fit encoders.
- Local app source and requirements already present in the existing repository were not modified in this documentation package.
- Never run cloud-related commands with published or hard-coded AWS credentials; use proper IAM profiles and stop unused paid resources.
