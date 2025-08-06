import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv('dataset.csv')
print("Data Loaded:", df.shape)

# Clean data
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Show basic info
print(df.info())
print(df.describe())

# ======= Audio Feature Distributions =======
features = ['danceability', 'energy', 'valence', 'acousticness']

for feature in features:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[feature], kde=True)
    plt.title(f"{feature.capitalize()} Distribution")
    plt.savefig(f"plots/{feature}_dist.png")
    plt.close()

# ======= Correlation Heatmap =======
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Between Features")
plt.savefig("plots/correlation_heatmap.png")
plt.close()

# ======= Popularity Over the Years =======
if 'year' not in df.columns:
    df['year'] = 2024

yearly_popularity = df.groupby('year')['popularity'].mean()
plt.figure(figsize=(10, 5))
yearly_popularity.plot()
plt.title("Average Popularity Over Years")
plt.xlabel("Year")
plt.ylabel("Popularity")
plt.savefig("plots/popularity_over_years.png")
plt.close()

# ======= Clustering Songs =======
X = df[['danceability', 'energy', 'valence']]
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

sns.pairplot(df, hue='cluster', vars=['danceability', 'energy', 'valence'])
plt.suptitle("K-Means Clustering of Songs", y=1.02)
plt.savefig("plots/clustering.png")
plt.close()

print("Analysis completed. Check the 'plots' folder for visualizations.")
import matplotlib.pyplot as plt
import seaborn as sns

features = ['danceability', 'energy', 'valence', 'acousticness']

for feature in features:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[feature], kde=True, color='green')
    plt.title(f"{feature.capitalize()} Distribution")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"plots/{feature}_dist.png")
    plt.close()
import seaborn as sns
import matplotlib.pyplot as plt

# Step 3: Correlation Heatmap
print("Generating correlation heatmap...")

# Select only numerical columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.close()

print("Correlation heatmap saved to plots/correlation_heatmap.png")
   

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 4: K-Means Clustering
print("Performing K-Means clustering...")

# Select features
features = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Visualize using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=df['cluster'], palette='viridis', alpha=0.6)
plt.title("K-Means Clustering Visualization (PCA)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("plots/kmeans_pca.png")
plt.close()

print("K-Means clustering plot saved to plots/kmeans_pca.png")
