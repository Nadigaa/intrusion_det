import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os

# Import engineered feature generator
sys.path.append(os.path.join(os.path.dirname(__file__)))
from feature_engineering import add_flow_features

# Load model + training feature names
model = joblib.load("models/zeek_attack_model.pkl")
features = joblib.load("models/zeek_features.pkl")

print("📌 Loading dataset...")
df = pd.read_csv("data/attack-add.csv", low_memory=False)

print("📌 Applying feature engineering...")
df = add_flow_features(df)

# Replace missing engineered values
df = df.fillna(0)

# Select features used during training
print("📌 Selecting ML features...")
df_model = df[features]

# Ground truth
y_true = df["attack_cat"]

print("📌 Running predictions...")
y_pred = model.predict(df_model)

# Create performance table
print("📌 Generating performance table...")

report = classification_report(y_true, y_pred, output_dict=True)
performance_df = pd.DataFrame(report).T

performance_df.to_csv("09_model_performance_table.csv", index=True)

print("\n🎉 Saved: 09_model_performance_table.csv")
print(performance_df)
