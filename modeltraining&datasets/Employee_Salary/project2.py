import pandas as pd

df = pd.read_csv(r"Dataset09-Employee-salary-prediction.csv")

#education level mein ordinal encoder lagana hai
#gender mein ohe lga dena hai
#age or experience mein standard scaler lga denge
#fill na values
#

df['Age'].fillna(df['Age'].mean() , inplace=True)
df['Gender'].fillna(df['Gender'].mode()[0] , inplace=True)
df['Education Level'].fillna(df['Education Level'].mode()[0] , inplace=True)
df['Job Title'].fillna(df['Job Title'].mode()[0] , inplace=True)
df['Years of Experience'].fillna(df['Years of Experience'].mean() , inplace=True)
df['Salary'].fillna(df['Salary'].mean() , inplace=True)