import pandas as pd

# -------------------------------------------------
# Load labelled mutation + risk dataset
# -------------------------------------------------
df = pd.read_csv("labelled_mutation_dataset.csv")

print("Original dataset shape:", df.shape)

# -------------------------------------------------
# Gene → Disease mapping (BIOLOGICALLY CORRECT)
# -------------------------------------------------
gene_disease_map = {
    "CFTR": "CF",   # Cystic Fibrosis
    "HBB": "SCD",   # Sickle Cell Disease
    "HTT": "HD"     # Huntington’s Disease
}

# -------------------------------------------------
# Assign disease based on gene
# -------------------------------------------------
df["Disease_Name"] = df["Gene"].map(gene_disease_map)

# Remove unrelated genes
df = df.dropna(subset=["Disease_Name"])

# -------------------------------------------------
# Encode disease labels for ML
# -------------------------------------------------
disease_encoding = {
    "CF": 0,
    "SCD": 1,
    "HD": 2
}

df["disease"] = df["Disease_Name"].map(disease_encoding)

# -------------------------------------------------
# Select final features for Random Forest
# -------------------------------------------------
final_df = df[
    [
        "Risk_Score",
        "Mutation_Status",
        "Clinical_Significance",
        "disease"
    ]
]

# Encode categorical columns
final_df["Mutation_Status"] = final_df["Mutation_Status"].map(
    {"No Mutation": 0, "Mutation": 1}
)

final_df["Clinical_Significance"] = final_df["Clinical_Significance"].map(
    {"Benign": 0, "Pathogenic": 1}
)

# -------------------------------------------------
# Save final training dataset
# -------------------------------------------------
final_df.to_csv("rf_training_dataset.csv", index=False)

print("✅ Final Random Forest training dataset created")
print(final_df["disease"].value_counts())
