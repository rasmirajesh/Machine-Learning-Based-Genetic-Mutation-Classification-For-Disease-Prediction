import pandas as pd

# Load dataset
df = pd.read_csv("labelled_mutation_dataset.csv")

print("✅ Dataset loaded")
print("Shape:", df.shape)

# -----------------------------
# Disease labeling logic
# -----------------------------
# Low risk + Benign → CF
# Medium risk → SCD
# High risk + Pathogenic → HD

def assign_disease(row):
    if row["Clinical_Significance"] == "Benign" and row["Risk_Score"] < 1.8:
        return 0   # CF
    elif row["Clinical_Significance"] == "Pathogenic" and row["Risk_Score"] > 2.5:
        return 2   # HD
    else:
        return 1   # SCD

df["disease"] = df.apply(assign_disease, axis=1)

# Encode categorical features
df["Mutation_Status"] = df["Mutation_Status"].map(
    {"No Mutation": 0, "Mutation": 1}
)

df["Clinical_Significance"] = df["Clinical_Significance"].map(
    {"Benign": 0, "Pathogenic": 1}
)

# Final dataset
final_df = df[
    ["Risk_Score", "Mutation_Status", "Clinical_Significance", "disease"]
]

# Save
final_df.to_csv("rf_training_dataset.csv", index=False)

print("✅ rf_training_dataset.csv created")
print("\nDisease distribution:")
print(final_df["disease"].value_counts())
