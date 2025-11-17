import pandas as pd
import json
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# =============================
# Load JSON Predictions
# =============================
with open("predictions_all.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(f"Loaded {df.shape[0]} total predictions.")

# Ensure columns exist
if "attack_type" not in df.columns or "attack_cat" not in df.columns:
    raise ValueError("JSON must contain 'attack_type' and 'attack_cat' fields.")

# =============================
# Clean labels for analysis
# =============================
y_true = df["attack_cat"].astype(str)
y_pred = df["attack_type"].astype(str)

# =============================
# Accuracy Report
# =============================
print("\n===== Classification Report =====")
print(classification_report(y_true, y_pred))

acc = accuracy_score(y_true, y_pred)
print(f"\nOverall Accuracy: {acc:.4f}")

# =============================
# Confusion Matrix
# =============================
cm = confusion_matrix(y_true, y_pred)
labels = sorted(df["attack_cat"].unique())

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

os.makedirs("analysis_plots", exist_ok=True)
plt.savefig("analysis_plots/confusion_matrix.png")
plt.close()

print("Saved: analysis_plots/confusion_matrix.png")

# =============================
# Class Distribution
# =============================
plt.figure(figsize=(8, 5))
df["attack_cat"].value_counts().plot(kind="bar", color="purple")
plt.title("True Class Distribution")
plt.xlabel("Attack Category")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("analysis_plots/true_class_distribution.png")
plt.close()

print("Saved: analysis_plots/true_class_distribution.png")

plt.figure(figsize=(8, 5))
df["attack_type"].value_counts().plot(kind="bar", color="green")
plt.title("Predicted Attack Distribution")
plt.xlabel("Predicted Category")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("analysis_plots/predicted_class_distribution.png")
plt.close()

print("Saved: analysis_plots/predicted_class_distribution.png")

print("\n🎉 Full analysis complete!")
