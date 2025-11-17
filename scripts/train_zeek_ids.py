import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from feature_engineering import add_flow_features

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

print("📌 Loading attack-add.csv...")
df = pd.read_csv("data/attack-add.csv", low_memory=False)

print("📌 Original shape:", df.shape)
print("\nattack_cat distribution:")
print(df["attack_cat"].value_counts())

# ==========================
# 1. REMOVE NON-USEFUL COLUMNS
# ==========================
drop_cols = [
    "uid", "id.orig_h_x", "id.resp_h_x", "method", "host", "uri",
    "referrer", "user_agent", "username", "password", "tags",
    "fuid", "filename", "sha1", "sha256", "md5",
    "extracted", "extracted_cutoff", "extracted_size"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ==========================
# 2. CLEAN NUMERIC COLUMNS
# ==========================
numeric_cols = [
    "duration_x", "orig_bytes", "resp_bytes",
    "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes",
    "trans_depth", "id.orig_p_x", "id.resp_p_x"
]

# Convert the basic numeric columns
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Total bytes (derived)
if "orig_bytes" in df.columns and "resp_bytes" in df.columns:
    df["total_bytes"] = df["orig_bytes"].fillna(0) + df["resp_bytes"].fillna(0)

# ==========================
# 3. ADD BEHAVIORAL FEATURES
# ==========================
df = add_flow_features(df)
print("📌 After feature engineering:", df.shape)

# ==========================
# 4. DROP ROWS WITH MISSING TARGET
# ==========================
df = df.dropna(subset=["attack_cat"])

# ==========================
# 5. SELECT FEATURES AND LABEL
# ==========================
y = df["attack_cat"]
X = df.drop(columns=["attack_cat", "label"], errors="ignore")

print("📌 Final ML feature set:", X.shape)

# ==========================
# 6. DEFINE NUMERIC & CATEGORICAL FEATURES
# ==========================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

# 🔥 FIX: Convert mixed-type columns to uniform strings
for col in categorical_features:
    X[col] = X[col].astype(str)


print("\nNumeric feature count:", len(numeric_features))
print("Categorical feature count:", len(categorical_features))

# ==========================
# 7. PREPROCESSING PIPELINES
# ==========================
numeric_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_tf, numeric_features),
        ("cat", categorical_tf, categorical_features),
    ]
)

# ==========================
# 8. MODEL PIPELINE
# ==========================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    class_weight="balanced_subsample",
    random_state=42
)

model = Pipeline(steps=[
    ("prep", preprocessor),
    ("clf", rf),
])

# ==========================
# 9. TRAIN/TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("🚀 Training model...")
model.fit(X_train, y_train)

# ==========================
# 10. EVALUATION
# ==========================
y_pred = model.predict(X_test)

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# ==========================
# 11. SAVE MODEL + FEATURE NAMES
# ==========================
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/zeek_attack_model.pkl")
joblib.dump(list(X.columns), "models/zeek_features.pkl")

print("\n🎉 Training complete!")
print("Saved:")
print(" - models/zeek_attack_model.pkl")
print(" - models/zeek_features.pkl")
