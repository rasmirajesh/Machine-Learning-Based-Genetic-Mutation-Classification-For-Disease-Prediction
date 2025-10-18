# inputdna.py
import pandas as pd
import numpy as np

# Load cleaned dataset to get the exact column structure
df_cleaned = pd.read_csv("cleaned_dataset.csv", index_col=0)

# Convert all columns to numeric and fill NaNs
df_cleaned = df_cleaned.apply(pd.to_numeric, errors='coerce').fillna(0)

np.random.seed(42)

# Option 1: No mutation (values close to mean)
mean_values = df_cleaned.mean(axis=0)
noise = np.random.normal(0, 0.01, size=mean_values.shape)  # tiny noise
sample_nomutation = pd.DataFrame([mean_values + noise], columns=df_cleaned.columns)
sample_nomutation.to_csv("input_dna_nomutation.csv", index=False)
print("Created input_dna_nomutation.csv (no mutation) with shape:", sample_nomutation.shape)

# Option 2: Random mutation (values randomly sampled)
sample_mutation = pd.DataFrame([np.random.rand(df_cleaned.shape[1])], columns=df_cleaned.columns)
sample_mutation.to_csv("input_dna_mutation.csv", index=False)
print("Created input_dna_mutation.csv (with mutation) with shape:", sample_mutation.shape)
