# EXPERIMENT 8 — PCA ON DIABETES DATASET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
data = pd.read_csv(r"C:\Users\ASTHA BAHETI\OneDrive\Documents\AIML\diabetes.csv")

print("Original Data:")
print(data.head())

# Separate features (X) and target (y)
X = data.drop('Outcome', axis=1)   # All columns except Output
y = data['Outcome']                # Diabetes result (0/1)

# Scale the features before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA - reduce to 2 components for plot
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create PCA DataFrame
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Outcome'] = y

print("\nAfter PCA Transformation:")
print(df_pca.head())

# Plot PCA result
plt.figure(figsize=(8, 6))
plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['Outcome'], cmap='coolwarm')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("2D PCA of Diabetes Dataset")
plt.colorbar(label="Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Show explained variance
print("\nExplained variance ratio (PC1, PC2):")
print(pca.explained_variance_ratio_)
print("Total variance explained:", sum(pca.explained_variance_ratio_))

# Full PCA - check cumulative variance
pca_full = PCA()
pca_full.fit(X_scaled)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
print("\nCumulative explained variance:")
print(cumulative_variance)

# Plot cumulative variance
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Variance Explained by PCA Components")
plt.grid(True)
plt.tight_layout()
plt.show()
