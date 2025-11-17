import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import os

# Import feature engineering module
sys.path.append(os.path.join(os.path.dirname(__file__)))
from feature_engineering import add_flow_features

# Load model + feature list
model = joblib.load("models/zeek_attack_model.pkl")
features = joblib.load("models/zeek_features.pkl")

print("📌 Loading dataset...")
df = pd.read_csv("data/attack-add.csv", low_memory=False)

print("📌 Applying feature engineering...")
df = add_flow_features(df)

# Ensure all missing engineered values become 0
df = df.fillna(0)

# Select the same ML features used during training
df_model = df[features]

print("📌 Predicting attack categories...")
preds = model.predict(df_model)

# Plot predicted distribution
plt.figure(figsize=(10,6))
sns.countplot(x=preds, palette="viridis")
plt.title("Predicted Attack Distribution (Feature Engineered Model)")
plt.xlabel("Predicted Category")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("08_predicted_distribution.png", dpi=150)
plt.show()

print("🎉 Saved plot: 08_predicted_distribution.png")
