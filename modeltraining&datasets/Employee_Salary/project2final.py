import pandas as pd

df = pd.read_csv(r"Cleaned Employee dataset.csv")
print(df.head())

print(df['Job Title'].value_counts())