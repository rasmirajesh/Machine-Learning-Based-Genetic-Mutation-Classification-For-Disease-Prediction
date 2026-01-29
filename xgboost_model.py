# ==========================================
# XGBOOST FOR GENETIC DISEASE CLASSIFICATION
# Classes: SCD, CF, HD
# Dataset: rf_training_dataset.csv
# ==========================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------
# Make sure rf_training_dataset.csv is in the same folder
data_path = "rf_training_dataset.csv"
df = pd.read_csv(data_path)

print("Dataset Shape:", df.shape)
print("Columns:", df.columns)

# ------------------------------------------
# 3. SPLIT FEATURES & LABEL
# ------------------------------------------
# Target column: 'disease'
X = df.drop("disease", axis=1)
y = df["disease"]

# ------------------------------------------
# 4. LABEL ENCODING (if disease is not already numeric)
# ------------------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nClass Mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(label, "->", i)

# ------------------------------------------
# 5. TRAIN–TEST SPLIT
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ------------------------------------------
# 6. DEFINE XGBOOST MODEL
# ------------------------------------------
model = XGBClassifier(
    objective="multi:softmax",
    num_class=3,                # SCD, CF, HD
    eval_metric="mlogloss",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# ------------------------------------------
# 7. TRAIN MODEL
# ------------------------------------------
print("\nTraining XGBoost model...")
model.fit(X_train, y_train)
print("Training completed.")

# ------------------------------------------
# 8. PREDICTION
# ------------------------------------------
y_pred = model.predict(X_test)

# ------------------------------------------
# 9. EVALUATION METRICS
# ------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["SCD", "CF", "HD"]   # FIXED ERROR HERE
))

# ------------------------------------------
# 10. CONFUSION MATRIX
# ------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["SCD", "CF", "HD"],
    yticklabels=["SCD", "CF", "HD"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("XGBoost Disease Classification (SCD / CF / HD)")
plt.tight_layout()
plt.show()

# ------------------------------------------
# 11. FEATURE IMPORTANCE
# ------------------------------------------
plt.figure(figsize=(8, 6))
plot_importance(model, max_num_features=10)
plt.title("Top 10 Important Features (XGBoost)")
plt.tight_layout()
plt.show()

# ------------------------------------------
# 12. SAVE MODEL (OPTIONAL)
# ------------------------------------------
model.save_model("xgboost_disease_model.json")
print("\nModel saved as xgboost_disease_model.json")

# ------------------------------------------
# 13. TEST ON NEW SAMPLE (OPTIONAL)
# ------------------------------------------
# Example sample - replace values with real feature values
sample = np.array([[0.5, 1, 0]])   # Must match number of feature columns
sample_pred = model.predict(sample)
predicted_label = ["SCD", "CF", "HD"][sample_pred[0]]

print("\nPrediction for new sample:", predicted_label)
