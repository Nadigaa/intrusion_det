import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------
# Load model and features
# -------------------------
model = joblib.load("models/zeek_attack_model.pkl")

# Extract preprocessing steps
preprocessor = model.named_steps["prep"]
rf = model.named_steps["clf"]

# -------------------------
# Get feature names from ColumnTransformer
# -------------------------
feature_names = []

# 1️⃣ Numeric features
num_features = preprocessor.named_transformers_["num"].named_steps["imputer"].feature_names_in_
feature_names.extend(num_features)

# 2️⃣ One-hot encoded categorical features
ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
cat_raw = preprocessor.named_transformers_["cat"].named_steps["imputer"].feature_names_in_
ohe_features = ohe.get_feature_names_out(cat_raw)
feature_names.extend(ohe_features)

print(f"Total extracted feature names: {len(feature_names)}")
print(f"Model expects: {rf.feature_importances_.shape[0]}")

# -------------------------
# Create DataFrame
# -------------------------
importances = rf.feature_importances_

df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

# Save full importance table
os.makedirs("results", exist_ok=True)
df.to_csv("results/feature_importance_full.csv", index=False)

# -------------------------
# Plot top 20
# -------------------------
topN = 20
top = df.head(topN)

plt.figure(figsize=(12, 8))
plt.barh(top["feature"], top["importance"])
plt.gca().invert_yaxis()
plt.title("Top 20 Most Important Features – Zeek IDS")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("results/feature_importance_top20.png", dpi=300)
plt.close()

print("🎉 Feature importance generated!")
print(" - results/feature_importance_full.csv")
print(" - results/feature_importance_top20.png")
