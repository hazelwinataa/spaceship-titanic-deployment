import os
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold

from ingest import load_data
from preprocess import fit_preprocessor, transform_data, save_preprocessor
from train import get_baseline_cv_score
from tune_logreg import tune_logistic_regression
from evaluate import evaluate_model


RANDOM_STATE = 42


def run_pipeline():
    print("=== LOAD DATA ===")
    train_df = load_data("data/train.csv")
    print(f"Train shape: {train_df.shape}")

    print("\n=== FIT PREPROCESSOR ===")
    preprocessor = fit_preprocessor(train_df)

    print("\n=== TRANSFORM DATA ===")
    X, y = transform_data(train_df, preprocessor, is_train=True)
    print(f"Processed feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    print("\n=== TRAIN / VALIDATION SPLIT ===")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"X_train: {X_train.shape}")
    print(f"X_val  : {X_val.shape}")

    print("\nBASELINE LOGISTIC REGRESSION CV")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    baseline_metrics = get_baseline_cv_score(X_train, y_train, cv=cv, random_state=RANDOM_STATE)
    print("Baseline CV Metrics:")
    for k, v in baseline_metrics.items():
        print(f"{k}: {v}")

    print("\nTUNING LOGISTIC REGRESSION")
    best_model, best_params, best_cv_score = tune_logistic_regression(
        X_train,
        y_train,
        cv=cv,
        n_trials=30,
        random_state=RANDOM_STATE
    )

    print("\nBest Params:")
    print(best_params)
    print(f"Best CV Accuracy: {best_cv_score:.4f}")

    print("\n=== VALIDATION EVALUATION ===")
    eval_results = evaluate_model(best_model, X_val, y_val)

    print("\n=== FIT FINAL MODEL ON FULL DATA ===")
    final_model, final_params, final_cv_score = tune_logistic_regression(
        X,
        y,
        cv=cv,
        n_trials=30,
        random_state=RANDOM_STATE
    )

    print("\nFinal Best Params:")
    print(final_params)
    print(f"Final CV Accuracy: {final_cv_score:.4f}")

    print("\n=== SAVE PICKLE FILES ===")
    os.makedirs("models", exist_ok=True)

    save_preprocessor(preprocessor, "models/preprocessor.pkl")

    with open("models/logreg_model.pkl", "wb") as f:
        pickle.dump(final_model, f)

    print("Saved: models/preprocessor.pkl")
    print("Saved: models/logreg_model.pkl")

    return {
        "baseline_metrics": baseline_metrics,
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "eval_results": eval_results,
        "final_params": final_params,
        "final_cv_score": final_cv_score
    }


if __name__ == "__main__":
    run_pipeline()

# ============================================================
# Penjelasan Singkat Kode Saya
# ============================================================
# File ini merupakan pipeline utama untuk menjalankan seluruh
# proses machine learning dari awal sampai akhir.

# Tahapan yang dilakukan dalam pipeline ini:
# 1. Load Data
#    Membaca dataset training dari file CSV.

# 2. Fit Preprocessor
#    Membuat preprocessor yang berisi feature engineering,
#    encoder untuk fitur kategorikal, serta nilai untuk
#    mengisi missing value.

# 3. Transform Data
#    Mengubah data mentah menjadi fitur numerik yang siap
#    digunakan oleh model.

# 4. Train / Validation Split
#    Membagi data menjadi data training dan validation
#    menggunakan stratified split.

# 5. Baseline Model Evaluation
#    Menghitung performa awal Logistic Regression menggunakan
#    cross-validation.

# 6. Hyperparameter Tuning
#    Menggunakan Optuna untuk mencari kombinasi parameter
#    terbaik untuk model Logistic Regression.

# 7. Validation Evaluation
#    Menguji performa model terbaik pada data validation.

# 8. Final Training
#    Melatih kembali model dengan parameter terbaik menggunakan
#    seluruh data training.

# 9. Save Model
#    Menyimpan preprocessor dan model Logistic Regression
#    yang sudah dioptimasi ke file pickle (.pkl).

# File ini memastikan seluruh proses training dapat dijalankan
# secara otomatis melalui satu pipeline.
# ============================================================