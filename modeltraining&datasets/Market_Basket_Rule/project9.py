import pandas as pd
from mlxtend.preprocessing.transactionencoder import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import joblib

# -----------------------------
# 1. Load and clean data
# -----------------------------
df = pd.read_csv(r'project9\Online Retail.csv')

df = df[['InvoiceNo', 'Description']]
df.dropna(inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype(str)

# -----------------------------
# 2. Prepare transactions
# -----------------------------
transactions = df.groupby('InvoiceNo')['Description'].apply(list).to_list()

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

basket = pd.DataFrame(te_array, columns=te.columns_)

# -----------------------------
# 3. Apply Apriori
# -----------------------------
frequent = apriori(basket, min_support=0.02, use_colnames=True)

rules = association_rules(frequent, min_threshold=1, metric='lift')
rules = rules.sort_values('lift', ascending=False)

# -----------------------------
# 4. Convert frozensets → comma strings
# -----------------------------
rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

# -----------------------------
# 5. Split multi-item consequents into separate rows
# -----------------------------
clean_rows = []
for _, row in rules.iterrows():
    antecedent = row["antecedents"]
    consequents = row["consequents"].split(",")

    for cons in consequents:
        clean_rows.append([
            antecedent.strip(),
            cons.strip(),
            row["support"],
            row["confidence"],
            row["lift"]
        ])

clean_df = pd.DataFrame(clean_rows, columns=["antecedents", "consequents", "support", "confidence", "lift"])

# -----------------------------
# 6. Save clean rules
# -----------------------------
joblib.dump(clean_df , 'Clean_Rules.pkl')