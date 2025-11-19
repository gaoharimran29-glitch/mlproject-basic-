import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

df = pd.read_csv("project10\symbipredict_2022.csv")

all_rules = []

for disease in df['prognosis'].unique():

    # filter rows of only THIS disease
    temp = df[df['prognosis'] == disease]

    # extract only symptoms (no disease column)
    symptom_cols = [c for c in df.columns if c != 'prognosis']

    # build transactions: symptoms only
    transactions = []
    for _, row in temp.iterrows():
        symptoms = [col for col in symptom_cols if row[col] == 1]
        transactions.append(symptoms)

    # one-hot encode
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    basket = pd.DataFrame(te_array, columns=te.columns_)

    # FP growth
    frequent = fpgrowth(basket, min_support=0.01, use_colnames=True , max_len=5)
    rules = association_rules(frequent, metric='confidence', min_threshold=0.6)

    # add disease label for clarity
    rules['Disease'] = disease

    # clean antecedents/consequents
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    rules = rules[['Disease', 'antecedents', 'consequents', 'support', 'confidence', 'lift']]
    rules = rules.sort_values(by='lift', ascending=False).head(5)  # top 3 rules
    

    all_rules.append(rules)

# merge for all diseases
final_rules = pd.concat(all_rules, ignore_index=True)
final_rules.to_csv("final_rules.csv", index=False)
print("Rules generated for all diseases!")