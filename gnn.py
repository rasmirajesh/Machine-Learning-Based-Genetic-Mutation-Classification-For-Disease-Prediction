# gnn_mutation_detection.py
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GAE, GCNConv
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------------------
# 1. Load & preprocess dataset
# ---------------------------
features = pd.read_csv("cleaned_dataset.csv", index_col=0)
print("Original features shape:", features.shape)

# Convert all non-numeric columns to numeric (if any)
features = features.apply(pd.to_numeric, errors='coerce').fillna(0)
print("Features shape after numeric conversion:", features.shape)

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features.values)
x = torch.tensor(features_scaled, dtype=torch.float)

# ---------------------------
# 2. Build top-k correlation graph
# ---------------------------
def build_topk_edges(df, k=5):
    corr = df.corr().abs()
    edges = []
    for i in range(corr.shape[0]):
        topk = corr.iloc[i].nlargest(k+1).index.tolist()  # include self
        for j in topk:
            if i != df.columns.get_loc(j):
                edges.append([i, df.columns.get_loc(j)])
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2,0), dtype=torch.long)
    return edge_index

edge_index = build_topk_edges(features, k=10)
print("Edge tensor shape:", edge_index.shape)

# ---------------------------
# 3. Define Graph Autoencoder
# ---------------------------
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

hidden_channels = 32
out_channels = 16
encoder = GCNEncoder(features.shape[1], hidden_channels, out_channels)
model = GAE(encoder)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 100

# ---------------------------
# 4. Training
# ---------------------------
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, edge_index)
    loss = model.recon_loss(z, edge_index)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ---------------------------
# 5. Generate embeddings & cluster
# ---------------------------
model.eval()
with torch.no_grad():
    embeddings = model.encode(x, edge_index).cpu().numpy()

# KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Save results
results = pd.DataFrame({
    "Sample": features.index,
    "Cluster": clusters
})
results.to_csv("mutation_clusters.csv", index=False)
print("Results saved! Sample:")
print(results.head())
