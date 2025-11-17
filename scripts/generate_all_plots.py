import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from feature_engineering import add_flow_features

os.makedirs("plots", exist_ok=True)

# -----------------------------------------------------
# 1. LOAD DATASET + APPLY FEATURE ENGINEERING
# -----------------------------------------------------
print("📌 Loading dataset...")
df = pd.read_csv("data/attack-add.csv", low_memory=False)

df = add_flow_features(df)
df = df.dropna(subset=["attack_cat"])

df["attack_cat"] = df["attack_cat"].astype(str)

print("Dataset:", df.shape)

# -----------------------------------------------------
# 2. LOAD MODEL + FEATURES
# -----------------------------------------------------
print("📌 Loading trained model...")
model = joblib.load("models/zeek_attack_model.pkl")
feature_list = joblib.load("models/zeek_features.pkl")

X = df[feature_list].fillna(0)
y = df["attack_cat"]
classes = sorted(y.unique())


# -----------------------------------------------------
# 3. PLOT: CLASS DISTRIBUTION
# -----------------------------------------------------
plt.figure(figsize=(8,5))
sns.countplot(x=y, palette="viridis")
plt.title("True Class Distribution")
plt.savefig("plots/01_class_distribution.png", dpi=300)
plt.close()


# -----------------------------------------------------
# 4. PLOT: CORRELATION HEATMAP (Top 15 features)
# -----------------------------------------------------
numeric_df = X.select_dtypes(include=[np.number])
corr = numeric_df.corr()

plt.figure(figsize=(12,10))
sns.heatmap(corr.iloc[:15, :15], annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap (Top 15 Features)")
plt.savefig("plots/02_correlation_heatmap.png", dpi=300)
plt.close()


# -----------------------------------------------------
# 5. SCATTER MATRIX (sample 2000 rows)
# -----------------------------------------------------
sample = df.sample(2000, random_state=42)
pair_features = ["flow_speed", "bytes_ratio", "pkts_ratio", "orig_bytes", "resp_bytes", "duration_x"]

sns.pairplot(sample[pair_features + ["attack_cat"]], hue="attack_cat", diag_kind="kde")
plt.savefig("plots/03_pairplot.png", dpi=300)
plt.close()


# -----------------------------------------------------
# 6. CONFUSION MATRIX
# -----------------------------------------------------
y_pred = model.predict(X)

cm = confusion_matrix(y, y_pred, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.savefig("plots/04_confusion_matrix.png", dpi=300)
plt.close()


# -----------------------------------------------------
# 7. CLASSIFICATION REPORT (as image)
# -----------------------------------------------------
from sklearn.metrics import classification_report
report = classification_report(y, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("plots/05_classification_report.csv")
print(report_df)


# -----------------------------------------------------
# 8. ROC CURVES (multiclass)
# -----------------------------------------------------
y_bin = label_binarize(y, classes=classes)
y_score = model.predict_proba(X)

plt.figure(figsize=(10,7))
for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve")
plt.legend()
plt.savefig("plots/06_multiclass_roc.png", dpi=300)
plt.close()


# -----------------------------------------------------
# 9. BEHAVIORAL FEATURE PLOTS
# -----------------------------------------------------
behavioral = ["flow_speed", "bytes_ratio", "pkts_ratio", "upload_bias", "download_bias"]

for feat in behavioral:
    plt.figure(figsize=(8,5))
    sns.boxplot(x=y, y=df[feat], palette="coolwarm")
    plt.title(f"{feat} per attack category")
    plt.xticks(rotation=45)
    plt.savefig(f"plots/07_box_{feat}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8,5))
    sns.histplot(data=df, x=feat, hue="attack_cat", element="step", stat="density")
    plt.title(f"{feat} distribution per attack type")
    plt.savefig(f"plots/08_hist_{feat}.png", dpi=300)
    plt.close()


# -----------------------------------------------------
# 10. FEATURE IMPORTANCE (already created)
# -----------------------------------------------------
print("Skipping feature importance (already generated).")

print("\n🎉 ALL PLOTS GENERATED → Check the /plots/ folder\n")
