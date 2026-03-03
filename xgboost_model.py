# # ==========================================
# # XGBOOST FOR GENETIC DISEASE CLASSIFICATION
# # Classes: SCD, CF, HD
# # Dataset: rf_training_dataset.csv
# # With SHAP Explainability
# # ==========================================

# # 1. IMPORT LIBRARIES
# import pandas as pd
# import numpy as np
# import shap

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# from xgboost import XGBClassifier, plot_importance
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ------------------------------------------
# # 2. LOAD DATASET
# # ------------------------------------------
# data_path = "rf_training_dataset.csv"
# df = pd.read_csv(data_path)

# print("Dataset Shape:", df.shape)
# print("Columns:", df.columns)

# # ------------------------------------------
# # 3. SPLIT FEATURES & LABEL
# # ------------------------------------------
# X = df.drop("disease", axis=1)
# y = df["disease"]

# # ------------------------------------------
# # 4. LABEL ENCODING
# # ------------------------------------------
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# print("\nClass Mapping:")
# for i, label in enumerate(label_encoder.classes_):
#     print(label, "->", i)

# # ------------------------------------------
# # 5. TRAIN–TEST SPLIT
# # ------------------------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y_encoded,
#     test_size=0.2,
#     random_state=42,
#     stratify=y_encoded
# )

# print("\nTraining samples:", X_train.shape[0])
# print("Testing samples:", X_test.shape[0])

# # ------------------------------------------
# # 6. DEFINE XGBOOST MODEL (SHAP-COMPATIBLE)
# # ------------------------------------------
# model = XGBClassifier(
#     objective="multi:softprob",   # REQUIRED FOR SHAP
#     num_class=3,
#     eval_metric="mlogloss",
#     n_estimators=300,
#     max_depth=6,
#     learning_rate=0.05,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42
# )

# # ------------------------------------------
# # 7. TRAIN MODEL
# # ------------------------------------------
# print("\nTraining XGBoost model...")
# model.fit(X_train, y_train)
# print("Training completed.")

# # ------------------------------------------
# # 8. PREDICTION (IMPORTANT FIX)
# # ------------------------------------------
# y_pred = model.predict(X_test)   # ← NO argmax here

# # ------------------------------------------
# # 9. EVALUATION METRICS
# # ------------------------------------------
# accuracy = accuracy_score(y_test, y_pred)
# print("\nAccuracy:", accuracy)

# print("\nClassification Report:")
# print(classification_report(
#     y_test,
#     y_pred,
#     target_names=["SCD", "CF", "HD"]
# ))

# # ------------------------------------------
# # 10. CONFUSION MATRIX
# # ------------------------------------------
# cm = confusion_matrix(y_test, y_pred)

# plt.figure(figsize=(6, 5))
# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     cmap="Blues",
#     xticklabels=["SCD", "CF", "HD"],
#     yticklabels=["SCD", "CF", "HD"]
# )
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("XGBoost Disease Classification (SCD / CF / HD)")
# plt.tight_layout()
# plt.show()

# # ------------------------------------------
# # 11. FEATURE IMPORTANCE
# # ------------------------------------------
# plt.figure(figsize=(8, 6))
# plot_importance(model, max_num_features=10)
# plt.title("Top 10 Important Features (XGBoost)")
# plt.tight_layout()
# plt.show()

# # ------------------------------------------
# # 12. SAVE MODEL
# # ------------------------------------------
# model.save_model("xgboost_disease_model.json")
# print("\nModel saved as xgboost_disease_model.json")

# # ------------------------------------------
# # 13. TEST ON NEW SAMPLE
# # ------------------------------------------
# sample = np.array([[0.5, 1, 0]])   # Risk_Score, Mutation_Status, Clinical_Significance
# sample_pred = model.predict(sample)
# predicted_label = ["SCD", "CF", "HD"][sample_pred[0]]

# print("\nPrediction for new sample:", predicted_label)

# # ==========================================
# # 14. SHAP EXPLAINABILITY (FINAL & WORKING)
# # ==========================================

# print("\nGenerating SHAP explanations...")

# # Use unified SHAP explainer (handles multi-class safely)
# explainer = shap.Explainer(model, X_train)

# # Compute SHAP values
# shap_values = explainer(X_test)

# # ------------------------------------------
# # 1️⃣ GLOBAL SHAP SUMMARY (ALL CLASSES)
# # ------------------------------------------
# shap.summary_plot(
#     shap_values,
#     X_test,
#     feature_names=X.columns
# )

# # ------------------------------------------
# # 2️⃣ CLASS-SPECIFIC SHAP (CF CLASS)
# # ------------------------------------------
# # Class index: 0 = SCD, 1 = CF, 2 = HD
# class_index = 1

# shap.summary_plot(
#     shap_values.values[:, :, class_index],
#     X_test,
#     feature_names=X.columns
# )

# # ------------------------------------------
# # 3️⃣ LOCAL EXPLANATION (ONE SAMPLE)
# # ------------------------------------------
# sample_index = 0

# shap.force_plot(
#     explainer.expected_value[class_index],
#     shap_values.values[sample_index, :, class_index],
#     X_test.iloc[sample_index],
#     matplotlib=True
# )


# ==========================================
# XGBOOST FOR GENETIC DISEASE CLASSIFICATION
# Classes: SCD, CF, HD
# ==========================================

# ==========================================
# XGBOOST FOR GENETIC DISEASE CLASSIFICATION
# Classes: SCD, CF, HD
# With Class Balancing + SHAP
# ==========================================

import os
import joblib
import pandas as pd
import numpy as np
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------
# 1️⃣ LOAD DATASET
# ------------------------------------------
data_path = "rf_training_dataset.csv"
df = pd.read_csv(data_path)

print("Dataset Shape:", df.shape)
print("Columns:", df.columns)

X = df.drop("disease", axis=1)
y = df["disease"]

feature_columns = X.columns.tolist()


# ------------------------------------------
# 2️⃣ LABEL ENCODING
# ------------------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nClass Mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{label} -> {i}")


# ------------------------------------------
# 3️⃣ TRAIN TEST SPLIT
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# ------------------------------------------
# 4️⃣ COMPUTE CLASS WEIGHTS (BALANCING)
# ------------------------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_encoded),
    y=y_encoded
)

class_weight_dict = dict(enumerate(class_weights))
print("\nClass Weights:", class_weight_dict)

sample_weights = np.array([class_weight_dict[label] for label in y_train])


# ------------------------------------------
# 5️⃣ DEFINE MODEL
# ------------------------------------------
model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    eval_metric="mlogloss",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


# ------------------------------------------
# 6️⃣ TRAIN MODEL WITH CLASS WEIGHTS
# ------------------------------------------
print("\nTraining XGBoost model with class balancing...")
model.fit(X_train, y_train, sample_weight=sample_weights)
print("Training completed.")


# ------------------------------------------
# 7️⃣ SAVE MODEL + ARTIFACTS
# ------------------------------------------
if not os.path.exists("backend"):
    os.makedirs("backend")

model_path = os.path.join("backend", "xgboost_disease_model.json")
encoder_path = os.path.join("backend", "label_encoder.pkl")
features_path = os.path.join("backend", "feature_columns.pkl")

model.save_model(model_path)
joblib.dump(label_encoder, encoder_path)
joblib.dump(feature_columns, features_path)

print("\nModel and artifacts saved successfully.")
print("Model ->", model_path)
print("Encoder ->", encoder_path)
print("Features ->", features_path)


# ------------------------------------------
# 8️⃣ EVALUATION
# ------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print(classification_report(
    y_test,
    y_pred,
    target_names=["SCD", "CF", "HD"]
))


# ------------------------------------------
# 9️⃣ CONFUSION MATRIX
# ------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("XGBoost Disease Classification")
plt.tight_layout()
plt.show()


# ------------------------------------------
# 🔟 SHAP EXPLAINABILITY (OPTIONAL)
# ------------------------------------------
print("\nGenerating SHAP explanations...")

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test)