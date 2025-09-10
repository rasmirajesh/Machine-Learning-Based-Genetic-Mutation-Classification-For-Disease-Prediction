import pandas as pd

# Load dataset (choose one)
df = pd.read_csv("combined_dataset.csv")   # change filename if needed

print("Original shape:", df.shape)
print(df.head())   # preview first rows

# --- Basic Cleaning Steps ---
# 1. Drop useless columns
drop_cols = ["GenBank Accession", "GO:Function", "GO:Process", 
             "GO:Component", "Unnamed: 1", "Unnamed: 2"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# 2. Handle missing Gene Symbols
if "Gene Symbol" in df.columns:
    df["Gene Symbol"] = df["Gene Symbol"].fillna(df["ID_REF"])

# 3. Group by Gene Symbol (mean expression)
if "Gene Symbol" in df.columns:
    df = df.groupby("Gene Symbol").mean().reset_index()

# 4. Transpose (samples as rows, genes as columns)
df_T = df.set_index("Gene Symbol").T

# 5. Save cleaned dataset
df_T.to_csv("cleaned_dataset.csv")
print("Cleaned dataset saved! Shape:", df_T.shape)
