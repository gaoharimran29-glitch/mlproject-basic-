import pandas as pd

df = pd.read_csv(r'project7\data.csv' , encoding='ISO-8859-1')

# Removing unnecessary columns
df.drop(columns=['StockCode' , 'Description'] , inplace=True)

# We have too much negative values in Quantity column which might indicate returns
# So we will remove negative values because we focus on study purchases instead of returns
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Changing datatype of InvoiceDate column because it is in object datatype
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Making customer id as int datatype
df = df[ pd.to_numeric(df['CustomerID'] , errors='coerce').notnull()]
df['CustomerID'] = df['CustomerID'].astype(int)

# Calculating Total Amount
df['TotalAmount'] = round(df['Quantity'] * df['UnitPrice'] , 2)
df.drop(columns=['Quantity' , 'UnitPrice'] , inplace=True)

print(df.info())

""" df.to_csv('Cleaned Customers Dataset.csv' , index=False) """