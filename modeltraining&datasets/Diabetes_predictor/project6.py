import pandas as pd
import numpy as np

df = pd.read_csv(r"diabetes.csv")

cols_with_zero_invalid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero_invalid] = df[cols_with_zero_invalid].replace(0 , np.nan)

for i in cols_with_zero_invalid:
    df[i].fillna(df[i].median() , inplace=True)

for col in ['Insulin', 'SkinThickness', 'BMI']:
    upper_limit = df[col].quantile(0.99)
    df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])

print(df['Outcome'].value_counts())

df.to_csv('cleaned_diabetes.csv' , index=False)