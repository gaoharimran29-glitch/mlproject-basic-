# Creating RFM table
""" RFM stands for:
R – Recency: How recently a customer made a purchase
F – Frequency: How often they made a purchase
M – Monetary: How much money they spent in total
It’s a method used in marketing, e-commerce, and customer analytics to find out:
Who are your loyal customers
Who are inactive customers
Who spends the most
Who may be about to leave """

import pandas as pd

df = pd.read_csv(r'Cleaned Customers Dataset.csv')

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

latest_date = df['InvoiceDate'].max()

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x : (latest_date - x.max()).days , #recency
    'InvoiceNo': 'nunique' , #frequency
    'TotalAmount' : 'sum' #monetary
})

rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalAmount': 'Monetary'
}, inplace=True)

rfm = rfm.reset_index()

rfm.to_csv('RFM data.csv' , index=False)