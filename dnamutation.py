# dnamutation.py

import pandas as pd
import numpy as np
import torch
from torch_geometric.nn import GAE, GCNConv
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.pyplot as plt

# ===========================
# 1. Load & preprocess dataset
# ===========================
features = pd.read_csv("cleaned_dataset.csv", index_col=0, low_memory=False)
features = features.apply(pd.to_numeric, errors='coerce').fillna(0)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features.values)

# Node features = reference gene profile
x_nodes = torch.tensor(
    features_scaled.mean(axis=0).reshape(-1, 1),
    dtype=torch.float
)
num_genes = x_nodes.shape[0]
print("Node feature shape:", x_nodes.shape)

# ===========================
# 2. Build gene-gene graph
# ===========================
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
print("Edge index shape:", edge_index.shape)

# ===========================
# 2a. Graph visualization (LABELLED)
# ===========================
G = nx.Graph()
G.add_nodes_from(range(num_genes))
G.add_edges_from(edge_index.t().tolist())

# ---- ADD LABELS (KEY FIX) ----
node_labels = {i: gene for i, gene in enumerate(features.columns)}

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42)

nx.draw(
    G,
    pos,
    with_labels=True,
    labels=node_labels,
    node_size=500,
    node_color='skyblue',
    edge_color='gray',
    font_size=8
)

plt.title("Gene–Gene Correlation Graph (Labelled)")
plt.show()

# ===========================
# 3. GCN Autoencoder
# ===========================
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

encoder = GCNEncoder(1, 32, 16)
model = GAE(encoder)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ===========================
# 4. Train GNN
# ===========================
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x_nodes, edge_index)
    loss = model.recon_loss(z, edge_index)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ===========================
# 5. Load input DNA sample
# ===========================
input_path = input("Enter path to DNA CSV file: ").strip()
input_dna = pd.read_csv(input_path)
input_dna = input_dna.apply(pd.to_numeric, errors='coerce').fillna(0)
input_dna = input_dna[features.columns]

input_scaled = scaler.transform(input_dna.values)

reference_mean = features_scaled.mean(axis=0)
deviation = input_scaled - reference_mean

# ===========================
# 6. Mutation detection
# ===========================
threshold = 2 * features_scaled.std(axis=0)
mutation_status = []

for i in range(deviation.shape[1]):
    if np.any(np.abs(deviation[:, i]) > threshold[i]):
        mutation_status.append("Mutation")
    else:
        mutation_status.append("No Mutation")

# ===========================
# 7. Mutation severity scoring
# ===========================
# Deviation magnitude
deviation_score = np.mean(np.abs(deviation), axis=0)

# Mutation frequency
mutation_frequency = np.mean(np.abs(deviation) > threshold, axis=0)

# GNN embedding importance
model.eval()
with torch.no_grad():
    z = model.encode(x_nodes, edge_index)
embedding_norm = torch.norm(z, dim=1).numpy()

# Graph connectivity score
degree_dict = dict(G.degree())
degree_score = np.array([degree_dict[i] for i in range(num_genes)])
degree_score = degree_score / degree_score.max()

# ===========================
# 8. Normalize & label
# ===========================
def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

risk_score = (
    normalize(deviation_score)
    + normalize(mutation_frequency)
    + normalize(embedding_norm)
    + normalize(degree_score)
)

# Adaptive pathogenic vs benign rule
cutoff = np.percentile(risk_score, 70)

labels = [
    "Pathogenic" if s >= cutoff else "Benign"
    for s in risk_score
]

# ===========================
# 9. Save labelled dataset
# ===========================
results = pd.DataFrame({
    "Gene": features.columns,
    "Mutation_Status": mutation_status,
    "Risk_Score": risk_score,
    "Clinical_Significance": labels
})

results.to_csv("labelled_mutation_dataset.csv", index=False)

print("\n✅ Labelled dataset saved as labelled_mutation_dataset.csv")
print(results["Clinical_Significance"].value_counts())
print(results.head())
