import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np

# Load dataset
df = pd.read_csv(r'project8/Cleaned House Data.csv')

df['house_age'] = 2025 - df['yr_built']
df['yr_since_renovated'] = 2025 - df['yr_renovated_fixed']
df['price'] = np.log1p(df['price'])
df['sqft_living'] = np.log1p(df['sqft_living'])
df['sqft_basement'] = np.log1p(df['sqft_basement'])
df['sqft_lot'] = np.log1p(df['sqft_lot'])
df['sqft_above'] = np.log1p(df['sqft_above'])
df['Total Rooms'] = df['bedrooms'] + df['bathrooms']
df['living ratio'] = df['sqft_living'] / df['sqft_lot']
df['basement_ratio'] = df['sqft_basement'] / df['sqft_living']
df['above_ratio'] = df['sqft_above'] / df['sqft_living']
df['price_per_sqft'] = df['price'] / df['sqft_living']

# Columns to use for clustering
features = ['floors',
       'waterfront', 'view', 'condition', 'grade', 'sqft_above',
       'house_age', 'yr_since_renovated' , 'Total Rooms' , 'living ratio' , 'basement_ratio' , 'above_ratio' , 'price_per_sqft']

# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[features])

# Apply PCA (reduce to 3 dimensions)
pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("Explained variance ratio by PCA components:", pca.explained_variance_ratio_)

# Define clustering models
models = {
    'KMeans': KMeans(random_state=42, n_clusters=4, init='k-means++' , n_init=20),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=4),  # eps reduced for PCA scaled data
    'OPTICS': OPTICS(min_samples=4 , eps=0.5),
    'Agglomerative Clustering': AgglomerativeClustering(n_clusters=4),
    'GMM': GaussianMixture(n_components=4, random_state=42)
}

# Run clustering
for name, model in models.items():
    print(f"\n{name}")
    if name == "GMM":
        cluster_labels = model.fit_predict(X_pca)
    else:
        cluster_labels = model.fit_predict(X_pca)
    
    if len(set(cluster_labels)) > 1:
        score = silhouette_score(X_pca, cluster_labels)
        print(f"Silhouette Score = {score:.3f}")
    else:
        print("Not enough clusters formed")

""" Explained variance ratio by PCA components: [0.4576726  0.26588037 0.07846767]

KMeans
Silhouette Score = 0.527

DBSCAN
Not enough clusters formed

OPTICS
Silhouette Score = -0.361

Agglomerative Clustering
Silhouette Score = 0.518

GMM
Silhouette Score = 0.417
""" 

# We are going to choose K-Means as it is giving 0.527
# A silhouette score of 0.531 represents the best practical clustering for this dataset, 
# balancing compact clusters with the continuous, overlapping nature of house features.