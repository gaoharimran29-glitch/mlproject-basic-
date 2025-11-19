import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
import joblib
import numpy as np

# Load cleaned dataset
df = pd.read_csv(r'project8\Cleaned House Data.csv')

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


# Columns for clustering
features = ['floors', 'waterfront', 'view', 'condition', 'grade', 'house_age',
       'yr_since_renovated', 'Total Rooms', 'living ratio', 'basement_ratio',
        'above_ratio','price_per_sqft']

X = df[features]

# Create pipeline
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('pca', PCA(n_components=3 , random_state=42)),
    ('kmeans', KMeans(n_clusters=4, random_state=42, n_init=20 , init='k-means++'))
])

# Fit the pipeline
pipeline.fit(X)

joblib.dump(pipeline, 'project8/house_clustering_pipeline.pkl')