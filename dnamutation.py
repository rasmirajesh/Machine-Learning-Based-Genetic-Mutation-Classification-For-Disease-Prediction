# dnamutation.py
import pandas as pd
import numpy as np
import torch
from torch_geometric.nn import GAE, GCNConv
from sklearn.preprocessing import StandardScaler

# ---------------------------
# 1. Load & preprocess dataset
# ---------------------------
features = pd.read_csv("cleaned_dataset.csv", index_col=0, low_memory=False)
features = features.apply(pd.to_numeric, errors='coerce').fillna(0)

# Scale features across samples
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features.values)  # [samples, genes]

# Node features = mean across samples (normal reference)
x_nodes = torch.tensor(features_scaled.mean(axis=0).reshape(-1, 1), dtype=torch.float)
num_genes = x_nodes.shape[0]
print("Node feature shape:", x_nodes.shape)

# ---------------------------
# 2. Build gene-gene correlation graph
# ---------------------------
def build_gene_edges(df, k=10):
    corr = df.corr().abs()
    edges = []
    for i in range(corr.shape[0]):
        topk_idx = corr.iloc[i].nlargest(k + 1).index
        for j in topk_idx:
            j_idx = corr.columns.get_loc(j)
            if i != j_idx:
                edges.append([i, j_idx])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

edge_index = build_gene_edges(features, k=10)
print("Edge tensor shape:", edge_index.shape)

# ---------------------------
# 3. Define GCN Autoencoder
# ---------------------------
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

encoder = GCNEncoder(in_channels=1, hidden_channels=32, out_channels=16)
model = GAE(encoder)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 50

# ---------------------------
# 4. Train GCN Autoencoder
# ---------------------------
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x_nodes, edge_index)
    loss = model.recon_loss(z, edge_index)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ---------------------------
# 5. Process input DNA
# ---------------------------
input_path = input("Enter path to your DNA CSV file: ").strip()
input_dna = pd.read_csv(input_path)
input_dna = input_dna.apply(pd.to_numeric, errors='coerce').fillna(0)
input_dna = input_dna[features.columns]

# Scale input DNA
input_scaled = scaler.transform(input_dna.values)

# Compute deviation from reference mean
reference_mean = features_scaled.mean(axis=0)
deviation = input_scaled - reference_mean

# ---------------------------
# 6. Determine mutation per gene
# ---------------------------
# Threshold: 2 standard deviations from normal
threshold = 2 * features_scaled.std(axis=0)
mutation_status = []
for i in range(deviation.shape[1]):
    # If any row in input DNA deviates more than threshold
    if np.any(np.abs(deviation[:, i]) > threshold[i]):
        mutation_status.append("Mutation")
    else:
        mutation_status.append("No Mutation")

# Save results
results = pd.DataFrame({
    "Gene": features.columns,
    "Mutation_Status": mutation_status
})
results.to_csv("dna_mutation_results.csv", index=False)
print("Results saved as dna_mutation_results.csv! Sample output:")
print(results.head())
