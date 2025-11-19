import pandas as pd

df = pd.read_csv(r"loan_sanction.csv")

#remove loan-id column
df.drop(columns="Loan_ID" , inplace=True)

#fill missing values
df['Gender'].fillna(df['Gender'].mode()[0] , inplace=True)
df['Married'].fillna(df['Married'].mode()[0] , inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0] , inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0] , inplace=True)
df['LoanAmount'].fillna(round(df['LoanAmount'].median() , 0) , inplace=True)
df['Loan_Amount_Term'].fillna(round(df['Loan_Amount_Term'].median()) , inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0] , inplace=True)
df['Loan_Status'].fillna(df['Loan_Status'].mode()[0] , inplace=True)
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

# Remove rows where both incomes are zero
df = df[~((df['ApplicantIncome'] == 0) & (df['CoapplicantIncome'] == 0))]
df.drop_duplicates(keep='first' ,inplace=True)
# Encode target column (Y -> 1, N -> 0)
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

#clean each columns
#change dependents columns type
print(df.columns)
print(df.isnull().sum())
print(df.info())
print(df.describe())

print(df['Loan_Status'].value_counts())

df.to_csv("Cleaned_Home_Loan.csv" , index=False)
