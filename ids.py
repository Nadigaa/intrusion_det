import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

print("📌 Loading datasets...")

df_attack = pd.read_csv("attack-add.csv", low_memory=False)
df_normal = pd.read_csv("Normal-trafic-4.csv", low_memory=False)

print("Attack rows:", df_attack.shape)
print("Normal rows:", df_normal.shape)

# Fill missing attack_cat for normal traffic
df_normal["attack_cat"] = "normal"

# Union of all columns
all_cols = sorted(list(set(df_attack.columns).union(df_normal.columns)))

df_attack = df_attack.reindex(columns=all_cols)
df_normal = df_normal.reindex(columns=all_cols)

# Merge
df = pd.concat([df_attack, df_normal], ignore_index=True)
print("\nMerged dataset:", df.shape)
print(df["label"].value_counts())

# Drop completely irrelevant columns if they exist
drop_cols = [
    "uid", "id.orig_h_x", "id.resp_h_x",
    "fuid", "parent_fuid", "md5", "sha1", "sha256",
    "host", "uri", "referrer", "user_agent",
]

drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=drop_cols)

# Replace NaN attack_cat
df["attack_cat"] = df["attack_cat"].fillna("normal")


# Define features and labels
X = df.drop(columns=["label", "attack_cat"])
y_bin = df["label"]
y_multi = df["attack_cat"]

# STEP 1 — Convert EVERYTHING to string (fixes mixed type issues)
X = X.astype(str)

# STEP 2 — Replace all NaN with a safe placeholder
X = X.fillna("missing")

# STEP 3 — Manually choose numeric columns (we control them)
numeric_cols = [
    "duration", "duration_x",
    "orig_bytes", "resp_bytes",
    "orig_pkts", "resp_pkts",
    "orig_ip_bytes", "resp_ip_bytes",
    "trans_depth",
    "id.orig_p_x", "id.resp_p_x",
    "id.orig_p_http", "id.resp_p_http",
    "ts", "ts_x", "ts_conn"
]

numeric_features = []
for col in numeric_cols:
    if col in X.columns:
        # convert to numeric
        X[col] = pd.to_numeric(X[col], errors="coerce")
        numeric_features.append(col)

# EVERYTHING else becomes categorical
categorical_features = [col for col in X.columns if col not in numeric_features]

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features[:10])
print("Total categorical:", len(categorical_features))

# Preprocessing pipeline
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# Split datasets
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)

# Models
rf_bin = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1)
rf_multi = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1)

bin_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", rf_bin),
])

multi_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", rf_multi),
])

# Train binary model
print("\n🚀 Training Binary IDS (normal vs attack)...")
bin_pipeline.fit(X_train_bin, y_train_bin)

y_pred_bin = bin_pipeline.predict(X_test_bin)
y_prob_bin = bin_pipeline.predict_proba(X_test_bin)[:, 1]

print("\n=== Binary IDS Results ===")
print(confusion_matrix(y_test_bin, y_pred_bin))
print(classification_report(y_test_bin, y_pred_bin))
print("ROC-AUC:", roc_auc_score(y_test_bin, y_prob_bin))

# Train multi-class model
print("\n🚀 Training Multi-Class Attack Model...")
multi_pipeline.fit(X_train_multi, y_train_multi)

y_pred_multi = multi_pipeline.predict(X_test_multi)

print("\n=== Multi-Class Attack Detection Results ===")
print(confusion_matrix(y_test_multi, y_pred_multi))
print(classification_report(y_test_multi, y_pred_multi))

# Save models
joblib.dump(bin_pipeline, "binary_ids_model.pkl")
joblib.dump(multi_pipeline, "multi_attack_model.pkl")

print("\n🎉 Training complete! Models saved:")
print(" - binary_ids_model.pkl")
print(" - multi_attack_model.pkl")
