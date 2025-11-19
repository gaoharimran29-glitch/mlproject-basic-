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

scaler = StandardScaler()

model = KMeans(n_clusters=4, random_state=42 , init='k-means++' , n_init=20)
X_scaled = scaler.fit_transform(X)

df['Cluster'] = model.fit_predict(X_scaled)


rfm_summary = df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
print(rfm_summary)

"""          Recency  Frequency   Monetary
Cluster
0          42.70       3.68    1359.05
1         247.08       1.55     480.62
2           6.38      82.54  127338.31
3          14.50      22.33   12709.09 """

cluster_labels = {
    0: "Potential Loyalist",
    1: "At Risk / Lost",
    2: "VIP / Loyal Customer",
    3: "Loyal Customer"
}

df['Segment'] = df['Cluster'].map(cluster_labels)
print(df[['CustomerID', 'Cluster', 'Segment']].head())


""" DBSCAN """
""" Silhouette Score = 0.839 """ 

""" KMeans
Silhouette Score = 0.616 """