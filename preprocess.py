import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Extract features from Cabin
    df["Deck"] = df["Cabin"].apply(
        lambda x: x.split("/")[0] if pd.notna(x) else "Unknown"
    )
    df["Cabin_num"] = df["Cabin"].apply(
        lambda x: x.split("/")[1] if pd.notna(x) else -1
    ).astype(float)
    df["Side"] = df["Cabin"].apply(
        lambda x: x.split("/")[2] if pd.notna(x) else "Unknown"
    )

    # Extract group information from PassengerId
    df["Group"] = df["PassengerId"].apply(
        lambda x: x.split("_")[0] if pd.notna(x) else "Unknown"
    )
    df["Group_size"] = df.groupby("Group")["Group"].transform("count")
    df["Solo"] = (df["Group_size"] == 1).astype(int)

    # Extract first name and last name from Name
    df["FirstName"] = df["Name"].apply(
        lambda x: x.split()[0] if pd.notna(x) else "Unknown"
    )
    df["LastName"] = df["Name"].apply(
        lambda x: x.split()[-1] if pd.notna(x) else "Unknown"
    )
    df["Family_size"] = df.groupby("LastName")["LastName"].transform("count")

    # Total spending features
    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpending"] = df[spending_cols].sum(axis=1)
    df["HasSpending"] = (df["TotalSpending"] > 0).astype(int)
    df["NoSpending"] = (df["TotalSpending"] == 0).astype(int)

    # Spending ratios
    for col in spending_cols:
        df[f"{col}_ratio"] = df[col] / (df["TotalSpending"] + 1)

    # Age groups
    df["Age_group"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 30, 50, 100],
        labels=["Child", "Teen", "Young_Adult", "Adult", "Senior"]
    )
    df["Age_group"] = df["Age_group"].astype(str)

    # Missing value indicators
    df["Age_missing"] = df["Age"].isna().astype(int)
    df["CryoSleep_missing"] = df["CryoSleep"].isna().astype(int)

    return df


def get_feature_lists(df: pd.DataFrame):
    categorical_features = [
        "HomePlanet", "CryoSleep", "Destination", "VIP",
        "Deck", "Side", "Age_group"
    ]

    numerical_features = [
        "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
        "Cabin_num", "Group_size", "Solo", "Family_size", "TotalSpending",
        "HasSpending", "NoSpending", "Age_missing", "CryoSleep_missing"
    ] + [col for col in df.columns if "_ratio" in col]

    return categorical_features, numerical_features


def fit_preprocessor(df: pd.DataFrame):
    df = df.copy()
    df = feature_engineering(df)

    categorical_features, numerical_features = get_feature_lists(df)

    fill_values_num = {}
    for col in numerical_features:
        if col in df.columns:
            fill_values_num[col] = df[col].median()

    label_encoders = {}
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
            le = LabelEncoder()
            le.fit(df[col])
            label_encoders[col] = le

    preprocessor = {
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "feature_columns": categorical_features + numerical_features,
        "fill_values_num": fill_values_num,
        "label_encoders": label_encoders
    }

    return preprocessor


def transform_data(df: pd.DataFrame, preprocessor: dict, is_train: bool = True):
    df = df.copy()
    df = feature_engineering(df)

    categorical_features = preprocessor["categorical_features"]
    numerical_features = preprocessor["numerical_features"]
    feature_columns = preprocessor["feature_columns"]
    fill_values_num = preprocessor["fill_values_num"]
    label_encoders = preprocessor["label_encoders"]

    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

            known_classes = set(label_encoders[col].classes_)
            df[col] = df[col].apply(lambda x: x if x in known_classes else "Unknown")

            if "Unknown" not in label_encoders[col].classes_:
                label_encoders[col].classes_ = list(label_encoders[col].classes_) + ["Unknown"]
                label_encoders[col].classes_ = pd.Index(label_encoders[col].classes_).astype(str).values

            df[col] = label_encoders[col].transform(df[col])

    for col in numerical_features:
        if col in df.columns:
            df[col] = df[col].fillna(fill_values_num[col])

    X = df[feature_columns]

    if is_train:
        y = df["Transported"].astype(int)
        return X, y
    else:
        return X


def save_preprocessor(preprocessor: dict, path: str):
    with open(path, "wb") as f:
        pickle.dump(preprocessor, f)


# ============================================================
# Penjelasan Singkat Kode Saya
# ============================================================
# File ini digunakan untuk melakukan preprocessing data sebelum
# data digunakan oleh model machine learning.

# Tahapan utama yang dilakukan:
# 1. Feature engineering:
#    Membuat fitur baru dari Cabin, PassengerId, Name, dan
#    fitur pengeluaran seperti TotalSpending dan spending ratio.

# 2. Menentukan daftar fitur:
#    Memisahkan fitur kategorikal dan numerik yang akan
#    digunakan oleh model.

# 3. Fit preprocessor:
#    Membuat LabelEncoder untuk fitur kategorikal dan
#    menghitung median untuk mengisi missing value fitur numerik.

# 4. Transform data:
#    Mengubah data mentah menjadi data numerik yang siap
#    digunakan model, termasuk encoding dan imputasi.

# 5. Save preprocessor:
#    Menyimpan object preprocessor ke file .pkl agar dapat
#    digunakan kembali saat deployment atau prediksi.
# ============================================================