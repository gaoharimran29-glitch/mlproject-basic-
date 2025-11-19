import pandas as pd

df1 = pd.read_csv(r"calories.csv")

df2 = pd.read_csv(r"exercise.csv")

df_merge = pd.merge(df1 , df2 , on="User_ID" , how='right')
df_merge.drop(columns="User_ID" , inplace=True)

df_merge.to_csv("Clean Calroies data.csv" , index=False)