from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def train_logistic_regression(X_train, y_train, random_state=42):
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000
    )
    model.fit(X_train, y_train)
    return model


def get_baseline_cv_score(X, y, cv, random_state=42):
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    return {
        "cv_accuracy_mean": float(scores.mean()),
        "cv_accuracy_std": float(scores.std()),
        "cv_accuracy_min": float(scores.min()),
        "cv_accuracy_max": float(scores.max()),
        "n_folds": int(cv.get_n_splits())
    }

# ============================================================
# Penjelasan Singkat Kode Saya
# ============================================================
# File ini digunakan untuk melakukan proses training model
# Logistic Regression serta evaluasi awal menggunakan
# cross-validation.

# Fungsi train_logistic_regression():
# - Membuat model Logistic Regression.
# - Melatih model menggunakan data training (X_train dan y_train).
# - Mengembalikan model yang sudah dilatih.

# Fungsi get_baseline_cv_score():
# - Menghitung performa awal model menggunakan cross-validation.
# - Menggunakan beberapa fold data untuk mengukur stabilitas model.
# - Menghasilkan beberapa metrik seperti:
#   • rata-rata accuracy
#   • standar deviasi accuracy
#   • nilai minimum dan maksimum accuracy
#   • jumlah fold yang digunakan.

# Hasil ini digunakan sebagai baseline performa model sebelum
# dilakukan tuning hyperparameter.
# ============================================================