# inputdna1.py
import pandas as pd
import numpy as np

# Step 1: Load training dataset
train = pd.read_csv("cleaned_dataset.csv", index_col=0, low_memory=False)
print("Original shape:", train.shape)

# Step 2: Convert all data to numeric safely
train_numeric = train.apply(pd.to_numeric, errors='coerce').fillna(0)

# Step 3: Add small noise (no mutation)
noise = np.random.uniform(0.98, 1.02, size=train_numeric.shape)
no_mutation_values = train_numeric.values * noise

# Step 4: Recreate dataframe with original columns and index
no_mutation_df = pd.DataFrame(no_mutation_values, index=train_numeric.index, columns=train_numeric.columns)

# Step 5: Save with same headers (so dnamutation.py recognizes columns)
no_mutation_df.to_csv("input_dna.csv")

print("✅ input_dna.csv (no mutation) created successfully with correct columns!")
print("Columns retained:", len(no_mutation_df.columns))
