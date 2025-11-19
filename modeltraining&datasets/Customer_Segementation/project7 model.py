import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(r'project7\RFM data.csv')

# Select features
X = df[['Recency', 'Frequency', 'Monetary']]

# Standardize
col_trans = make_column_transformer(
    (StandardScaler(), ['Recency', 'Frequency', 'Monetary']),
    remainder='passthrough'
)

# Store models
models = {
    "KMeans": KMeans(n_clusters=4, random_state=42 , init='k-means++' , n_init=20),
    "Agglomerative": AgglomerativeClustering(n_clusters=4),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "GMM": GaussianMixture(n_components=4, random_state=42)
}

# Fit & evaluate
for name, model in models.items():
    pipe = make_pipeline(col_trans, model)
    
    # Special case for GMM (it doesn’t have .fit_predict)
    if name == "GMM":
        X_scaled = StandardScaler().fit_transform(X)
        cluster_labels = model.fit_predict(X_scaled)
    else:
        cluster_labels = pipe.fit_predict(X)
    
    print(f"\n{name}")
    if len(set(cluster_labels)) > 1:
        X_scaled = col_trans.fit_transform(X)
        score = silhouette_score(X_scaled, cluster_labels)
        print(f"Silhouette Score = {score:.3f}")
    else:
        print("Not enough clusters formed")

""" DBSCAN """
""" Silhouette Score = 0.839 """ 

""" KMeans
Silhouette Score = 0.616 """