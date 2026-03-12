import optuna
from optuna.samplers import TPESampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def objective_lr(trial, X, y, cv, random_state=42):
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    solver = trial.suggest_categorical("solver", ["liblinear", "saga"])

    # valid combinations
    if solver == "liblinear" and penalty not in ["l1", "l2"]:
        raise optuna.exceptions.TrialPruned()

    if solver == "saga" and penalty not in ["l1", "l2"]:
        raise optuna.exceptions.TrialPruned()

    params = {
        "C": trial.suggest_float("C", 0.001, 100, log=True),
        "penalty": penalty,
        "solver": solver,
        "max_iter": trial.suggest_int("max_iter", 100, 2000),
        "random_state": random_state
    }

    model = LogisticRegression(**params)

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    return float(scores.mean())


def tune_logistic_regression(X, y, cv, n_trials=30, random_state=42):
    sampler = TPESampler(seed=random_state)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="logistic_regression_optimization"
    )

    study.optimize(
        lambda trial: objective_lr(trial, X, y, cv, random_state=random_state),
        n_trials=n_trials,
        show_progress_bar=True
    )

    best_params = study.best_params.copy()
    best_params["random_state"] = random_state

    best_model = LogisticRegression(**best_params)
    best_model.fit(X, y)

    return best_model, study.best_params, float(study.best_value)

# ============================================================
# Penjelasan Singkat Kode Saya
# ============================================================
# File ini digunakan untuk melakukan hyperparameter tuning
# pada model Logistic Regression menggunakan library Optuna.

# Fungsi objective_lr():
# - Menentukan ruang pencarian hyperparameter seperti:
#   penalty, solver, nilai C, dan max_iter.
# - Melatih model Logistic Regression dengan parameter yang
#   diusulkan oleh Optuna.
# - Menggunakan cross-validation untuk menghitung accuracy
#   rata-rata sebagai nilai evaluasi (objective value).

# Fungsi tune_logistic_regression():
# - Membuat studi Optuna untuk mencari kombinasi hyperparameter
#   terbaik menggunakan metode TPESampler.
# - Menjalankan proses tuning sebanyak n_trials percobaan.
# - Mengambil parameter terbaik yang ditemukan.
# - Melatih kembali model Logistic Regression menggunakan
#   parameter terbaik tersebut pada seluruh data training.

# Fungsi ini mengembalikan:
# - best_model : model Logistic Regression terbaik
# - best_params: parameter terbaik hasil tuning
# - best_value : nilai accuracy terbaik dari proses tuning
# ============================================================