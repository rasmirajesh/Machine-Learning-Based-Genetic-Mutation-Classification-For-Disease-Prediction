import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load features
features = pd.read_csv("features.csv", index_col=0)

# Drop columns where all values are NaN
features = features.dropna(axis=1, how='all')

# Fill remaining NaNs with column mean
features = features.fillna(features.mean())

# Convert to numeric (in case of string/object types)
features = features.apply(pd.to_numeric, errors='coerce')

# Clip extreme values to [-5, 5] to avoid numerical instability
features_scaled = StandardScaler().fit_transform(features)
features_scaled = np.clip(features_scaled, -5, 5)

# Check for remaining NaNs
if np.isnan(features_scaled).any():
    print("Warning: NaNs still present in features!")
else:
    print("Features cleaned. No NaNs remain.")
