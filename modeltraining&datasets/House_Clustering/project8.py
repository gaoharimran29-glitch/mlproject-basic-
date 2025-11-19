import pandas as pd
import numpy as np

df = pd.read_csv(r'project11\kc_house_data.csv')
df.drop(columns=['id' , 'date' , 'lat' , 'long' , 'zipcode' , 'sqft_living15' , 'sqft_lot15'] , inplace=True)
df = df[df['bedrooms']<=10]
df['yr_renovated_fixed'] = np.where(df['yr_renovated'] == 0, df['yr_built'], df['yr_renovated'])
df.drop(columns=['yr_renovated'] , inplace=True)


df.to_csv('project11/Cleaned House Data.csv' , index=False)