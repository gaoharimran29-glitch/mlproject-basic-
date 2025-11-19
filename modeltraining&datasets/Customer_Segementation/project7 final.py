import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import joblib

df = pd.read_csv(r'project7\RFM data.csv')

X = df[['Recency', 'Frequency', 'Monetary']]

col_trans = make_column_transformer(
    (StandardScaler() , ['Recency', 'Frequency', 'Monetary']) ,
    remainder='passthrough'
)

model = KMeans(n_clusters=4, random_state=42 , init='k-means++' , n_init=20)
pipe = make_pipeline(col_trans , model)

pipe.fit(X)

joblib.dump(pipe , 'customer_segmentation.pkl')