import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Load dataset
df = pd.read_csv(r'project10\symbipredict_2022.csv')
df = df.drop_duplicates()  # remove duplicate rows

all_rules = []

for disease in df['prognosis'].unique():
    disease_df = df[df['prognosis'] == disease]
    
    symptom_cols = disease_df.columns[-2:]
    symptom_cols = [col for col in symptom_cols if disease_df[col].notna().any()]  # remove empty columns
    
    if not symptom_cols:
        continue  # skip if no symptoms

    # Convert each row to a list of symptoms
    transactions = disease_df[symptom_cols].apply(
        lambda x: [sym.strip() for sym in x if pd.notna(sym) and sym.strip() != ''],
        axis=1
    ).to_list()
    
    # Remove exact duplicate transactions
    transactions = [list(x) for x in set(tuple(tx) for tx in transactions)]

    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    basket = pd.DataFrame(te_array, columns=te.columns_)

    # FP-Growth with very low support
    frequent = fpgrowth(basket, min_support=0.001, use_colnames=True, max_len=5)
    
        # Generate association rules
    rules = association_rules(frequent, metric='lift', min_threshold=0.6)
    if rules.empty:
        continue
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    rules['Disease'] = disease
    rules = rules[['Disease', 'antecedents', 'consequents', 'support', 'confidence', 'lift']]
    rules = rules.sort_values(by='lift', ascending=False).head(3)  # top 3 rules
    
    all_rules.append(rules)

# Combine all rules
final_rules = pd.concat(all_rules, ignore_index=True)
final_rules.to_csv('Disease_Rules.csv', index=False)
print("Rules generated for all diseases!")
