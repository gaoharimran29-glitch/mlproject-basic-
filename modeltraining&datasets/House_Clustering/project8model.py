import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
import joblib
import numpy as np

# Load cleaned dataset
df = pd.read_csv(r'project11\Cleaned House Data.csv')

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
df['price_per_sqft'] = df['price'] / df['sqft_living']


# Columns for clustering
features = ['floors', 'waterfront', 'view', 'condition', 'grade', 'house_age',
       'yr_since_renovated', 'Total Rooms', 'living ratio', 'basement_ratio',
       'price_per_sqft']

X = df[features]

# Create pipeline
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('pca', PCA(n_components=3 , random_state=42)),
    ('kmeans', KMeans(n_clusters=4, random_state=42, n_init=20 , init='k-means++'))
])

# Fit the pipeline
pipeline.fit(X)

# Transform the data and get cluster labels
X_pca = pipeline.named_steps['pca'].transform(pipeline.named_steps['scaler'].transform(X))
labels = pipeline.named_steps['kmeans'].labels_

""" score = silhouette_score(X_pca, labels)
print(f"Silhouette Score = {score:.3f}")
 """

""" f['cluster'] = labels
cluster_summary = df.groupby('cluster')[['price', 'living ratio']].mean()
print(cluster_summary)
 """

""" cluster   price         living ratio
0        13.006212      0.840526
1        13.167651      0.888555
2        12.845298      0.808201
3        13.646776      0.847163 """

""" cluster_labels = {
    0: "Medium Expensive House",
    1: "Medium-Large House",
    2: "Compact / Affordable House",
    3: "Premium House"
} """
