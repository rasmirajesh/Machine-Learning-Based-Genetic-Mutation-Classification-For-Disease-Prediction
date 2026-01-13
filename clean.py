import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---- Load dataset ----
df = pd.read_csv("combined_dataset.csv", low_memory=False)

# Identify expression and metadata columns
expression_cols = [col for col in df.columns if col.startswith("GSM")]
metadata_cols = [col for col in df.columns if col not in expression_cols]

# ---- Expression Data ----
# Rows = genes, Columns = samples
expression_data = df[expression_cols].apply(pd.to_numeric, errors="coerce")

# ---- Log2 + Standardize ----
expression_data = np.log2(expression_data + 1)
scaler = StandardScaler()
# Transpose: samples = rows (nodes), genes = features
expression_scaled = pd.DataFrame(
    scaler.fit_transform(expression_data.T),
    index=expression_data.columns,    # sample IDs (nodes)
    columns=df["ID_REF"]              # gene IDs (features)
)

# ---- Metadata ----
metadata = df[metadata_cols]  # now properly defined

# ---- Labels ----
labels = None
if "Mutation_Status" in metadata.columns:
    # Ensure Mutation_Status aligns with GSM sample IDs
    sample_meta = metadata.set_index("ID_REF") if "ID_REF" in metadata.columns else metadata
    if "Mutation_Status" in sample_meta.columns:
        labels = sample_meta["Mutation_Status"].map({"Yes": 1, "No": 0})
        labels = labels.reindex(expression_scaled.index)  # align with GSM IDs

# ---- Graph Edges (sample correlation) ----
corr = expression_scaled.T.corr()
edges = []
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        if corr.iloc[i, j] > 0.8:  # similarity threshold
            edges.append((corr.columns[i], corr.columns[j]))

edge_df = pd.DataFrame(edges, columns=["Source", "Target"])

# ---- Save outputs ----
expression_scaled.to_csv("features.csv")   # Node features
if labels is not None:
    labels.to_csv("labels.csv")           # Node labels
edge_df.to_csv("edges.csv", index=False)  # Graph structure

print("✅ GNN-ready dataset created: features.csv, labels.csv, edges.csv")
