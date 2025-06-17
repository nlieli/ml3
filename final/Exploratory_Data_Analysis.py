import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import warnings
warnings.filterwarnings('ignore')

# Konfiguration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Funktion zum Speichern der Plots
def save_plot(filename, dpi=300, bbox_inches='tight'):
    """Speichert den aktuellen Plot im OUTPUT_DIR"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, facecolor='white')
    print(f"Plot gespeichert: {filepath}")

# Daten laden
print("Loading data...")
df = pd.read_csv("D.csv")
df_out = pd.read_csv("D_out.csv").iloc[:, 1:]

# Features und Labels trennen
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X = X.drop(columns=['id'])  # Entfernen der 'id'-Spalte, falls vorhanden


print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Outlier dataset: {df_out.shape[0]} samples")

# 1. Class Distribution
print("\n1. Creating class distribution...")
plt.figure(figsize=(10, 6))
colors = sns.color_palette("Set2", len(np.unique(y)))
sns.countplot(x=y, palette=colors)
plt.title('Target Variable Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Class', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Werte über den Balken anzeigen
for i, count in enumerate(y.value_counts().sort_index()):
    plt.text(i, count + 0.5, str(count), ha='center', fontweight='bold')

plt.grid(axis='y', alpha=0.3)
save_plot("01_class_distribution.png")
plt.show()

# Klassenstatistiken ausgeben
for i in range(len(np.unique(y))):
    n = np.count_nonzero(y == i)
    print(f"Count of label {i}: {n}")

# 2. Correlation Matrix (improved and smaller)
print("\n2. Creating correlation matrix...")
corr = X.corr()
plt.figure(figsize=(10, 8))  # Reduced size

# Mask for upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Heatmap with correlation values
sns.heatmap(corr, 
            mask=mask, 
            annot=True,  # Show correlation values
            fmt='.2f',   # Format to 2 decimal places
            center=0, 
            vmax=1, 
            vmin=-1,
            cmap='RdBu_r',  # Improved color scheme
            square=True, 
            linewidths=0.5, 
            cbar_kws={"shrink": .6, "label": "Correlation Coefficient"},
            annot_kws={"size": 7})  # Smaller font size

plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
save_plot("02_correlation_matrix.png")
plt.show()

# 3. Standardize data
print("\n3. Standardizing data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. PCA Visualization
print("\n4. Creating PCA visualization...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], 
                         hue=y, 
                         palette="Set1", 
                         s=15,  # Much smaller points
                         alpha=0.7,
                         edgecolor='none')
plt.title(f"PCA – First Two Principal Components\n(Explained Variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%})", 
          fontsize=14, fontweight='bold')
plt.xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.1%} Variance)", fontsize=12)
plt.ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.1%} Variance)", fontsize=12)
plt.legend(title='Class', title_fontsize=12, fontsize=10)
plt.grid(True, alpha=0.3)
save_plot("03_pca_visualization.png")
plt.show()

# 5. t-SNE Visualization
print("\n5. Creating t-SNE visualization...")
tsne = TSNE(n_components=2, random_state=42, perplexity=50, learning_rate=200, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], 
               hue=y, 
               palette="Set1", 
               s=15,  # Much smaller points
               alpha=0.7,
               edgecolor='none')
plt.title("t-SNE Visualization", fontsize=14, fontweight='bold')
plt.xlabel("t-SNE 1", fontsize=12)
plt.ylabel("t-SNE 2", fontsize=12)
plt.legend(title='Class', title_fontsize=12, fontsize=10)
plt.grid(True, alpha=0.3)
save_plot("04_tsne_visualization.png")
plt.show()

# 6. UMAP Visualization
print("\n6. Creating UMAP visualization...")
reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], 
               hue=y, 
               palette="Set1", 
               s=15,  # Much smaller points
               alpha=0.7,
               edgecolor='none')
plt.title("UMAP Projection of Sensor Data", fontsize=14, fontweight='bold')
plt.xlabel("UMAP-1", fontsize=12)
plt.ylabel("UMAP-2", fontsize=12)
plt.legend(title='Class', title_fontsize=12, fontsize=10)
plt.grid(True, alpha=0.3)
save_plot("05_umap_visualization.png")
plt.show()

# 7. Feature Scatter Matrix (first 4 features) - FIXED
print("\n7. Creating feature scatter matrix...")
n_features = min(4, X.shape[1])
fig, axes = plt.subplots(n_features, n_features, figsize=(16, 16))
fig.suptitle('Feature Scatter Matrix', fontsize=16, fontweight='bold')

for i in range(n_features):
    for j in range(n_features):
        if i == j:
            # Diagonal: Histogram - FIXED: Use .loc instead of .iloc with boolean mask
            for label in np.unique(y):
                mask = y == label
                # Use .loc with boolean mask or convert to numpy array
                data_subset = X.iloc[:, i][mask]  # This is the fix
                axes[i, j].hist(data_subset, alpha=0.7, label=f'Class {label}', bins=20)
            axes[i, j].set_xlabel(X.columns[i])
            axes[i, j].legend()
        else:
            # Off-diagonal: Scatter plot
            scatter = axes[i, j].scatter(X.iloc[:, j], X.iloc[:, i], 
                                       c=y, cmap='Set1', alpha=0.6, s=10)
            axes[i, j].set_xlabel(X.columns[j])
            axes[i, j].set_ylabel(X.columns[i])
        axes[i, j].grid(True, alpha=0.3)

plt.tight_layout()
save_plot("06_feature_scatter_matrix.png")
plt.show()

# 8. Boxplots for first 6 features
print("\n8. Creating boxplots...")
n_features_box = min(6, X.shape[1])
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Feature Boxplots by Class', fontsize=16, fontweight='bold')

for i in range(n_features_box):
    row = i // 3
    col = i % 3
    
    sns.boxplot(x=y, y=X.iloc[:, i], ax=axes[row, col], palette="Set2")
    axes[row, col].set_title(f'Feature: {X.columns[i]}', fontweight='bold')
    axes[row, col].set_xlabel('Class')
    axes[row, col].set_ylabel(X.columns[i])
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
save_plot("07_boxplots_features.png")
plt.show()

# 9. Normal vs. Outlier Data
print("\n9. Creating normal vs. outlier visualization...")
X_out_scaled = scaler.transform(df_out)
X_all = np.vstack([X_scaled, X_out_scaled])
labels = ['Normal'] * len(X) + ['Outlier'] * len(df_out)

pca_all = PCA(n_components=2)
X_pca_all = pca_all.fit_transform(X_all)

plt.figure(figsize=(12, 8))
sns.scatterplot(x=X_pca_all[:,0], y=X_pca_all[:,1], 
               hue=labels, 
               palette=["#2E86AB", "#F24236"], 
               s=15,  # Much smaller points
               alpha=0.7,
               edgecolor='none')
plt.title("PCA: Normal vs. Outlier Data", fontsize=14, fontweight='bold')
plt.xlabel(f"PC 1 ({pca_all.explained_variance_ratio_[0]:.1%} Variance)", fontsize=12)
plt.ylabel(f"PC 2 ({pca_all.explained_variance_ratio_[1]:.1%} Variance)", fontsize=12)
plt.legend(title='Data Type', title_fontsize=12, fontsize=10)
plt.grid(True, alpha=0.3)
save_plot("08_normal_vs_outlier.png")
plt.show()

# 10. Descriptive Statistics
print("\n10. Descriptive Statistics:")
print("="*50)
print(df.describe())

# Summary
print(f"\n{'='*50}")
print("SUMMARY:")
print(f"{'='*50}")
print(f"All plots saved in '{OUTPUT_DIR}' folder!")
print(f"Number of plots created: 8")
print(f"File format: PNG with 300 DPI")
print(f"{'='*50}")
