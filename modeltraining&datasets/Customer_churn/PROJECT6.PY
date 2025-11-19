import pandas as pd
import numpy as np

df = pd.read_csv(r'WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.drop(columns='customerID' , inplace=True)

cols = ['OnlineSecurity' , 'OnlineBackup' ,'DeviceProtection' , 'TechSupport' , 'StreamingTV' , 'StreamingMovies']

for i in cols:
    df[i] = df[i].replace({'No internet service':'No'})

df['MultipleLines']=df['MultipleLines'].replace({'No phone service':'No'})

cols2 = ['OnlineSecurity' , 'OnlineBackup' ,'DeviceProtection' , 'TechSupport' , 'StreamingTV' , 'StreamingMovies' , 'MultipleLines' , 'PaperlessBilling' , 'Churn' , 'Partner' , 'Dependents' , 'PhoneService']

for i in cols2:
    df[i]= df[i].map({'Yes':1 , 'No':0})

for i in df.columns:
    print(df[i].unique())

df['TotalCharges'].replace(' ' , np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].mean() , inplace=True)
df['TotalCharges'].astype(float)



df.to_csv('Clean_Customer_Churn.csv' , index=False)