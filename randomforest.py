import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------------------------
# 1. Load RF training dataset
# -------------------------------------------------
df = pd.read_csv("rf_training_dataset.csv")

print("✅ Dataset loaded")
print("Shape:", df.shape)
print("\nDisease distribution:")
print(df["disease"].value_counts())

# -------------------------------------------------
# 2. Split features and target
# -------------------------------------------------
X = df.drop("disease", axis=1)
y = df["disease"]

# -------------------------------------------------
# 3. Train-test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# 4. Train Random Forest
# -------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

rf.fit(X_train, y_train)
print("\n✅ Random Forest training completed")

# -------------------------------------------------
# 5. Predictions
# -------------------------------------------------
y_pred = rf.predict(X_test)

# -------------------------------------------------
# 6. Evaluation
# -------------------------------------------------
print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------------------------
# 7. Feature Importance
# -------------------------------------------------
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n🔬 Feature Importance:")
print(importance)

plt.figure(figsize=(6, 4))
plt.barh(importance["Feature"], importance["Importance"])
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# -------------------------------------------------
# 8. Map labels to disease names
# -------------------------------------------------
label_to_disease = {
    0: "Cystic Fibrosis (CF)",
    1: "Sickle Cell Disease (SCD)",
    2: "Huntington’s Disease (HD)"
}

# -------------------------------------------------
# 9. Save predictions with disease names
# -------------------------------------------------
results = X_test.copy()
results["Actual_Label"] = y_test.values
results["Predicted_Label"] = y_pred

results["Actual_Disease"] = results["Actual_Label"].map(label_to_disease)
results["Predicted_Disease"] = results["Predicted_Label"].map(label_to_disease)

results.to_csv("rf_predictions_named.csv", index=False)

print("\n🧬 Sample Disease Classification Output:")
#print(results[["Actual_Disease", "Predicted_Disease"]].head())
print(results["Actual_Disease"].value_counts())
print(results["Predicted_Disease"].value_counts())


print("\n💾 Predictions saved to rf_predictions_named.csv")
print("\n✅ Random Forest disease classification completed successfully")
