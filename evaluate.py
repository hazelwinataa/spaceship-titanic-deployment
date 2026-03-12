from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)

    print("\n=== EVALUATION RESULT ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    results = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm
    }

    return results

# ============================================================
# Penjelasan Singkat Kode Saya
# ============================================================
# File ini digunakan untuk melakukan evaluasi performa model
# machine learning setelah proses training selesai.

# Fungsi evaluate_model():
# 1. Menggunakan model untuk melakukan prediksi pada data validasi (X_val).
# 2. Menghitung akurasi model menggunakan accuracy_score.
# 3. Menampilkan classification report yang berisi precision,
#    recall, dan F1-score untuk setiap kelas.
# 4. Menampilkan confusion matrix untuk melihat jumlah prediksi
#    benar dan salah pada setiap kelas.

# Hasil evaluasi kemudian disimpan dalam dictionary yang berisi:
# - accuracy
# - classification_report
# - confusion_matrix

# Dictionary ini dapat digunakan kembali untuk analisis
# performa model atau dicatat dalam pipeline.
# ============================================================