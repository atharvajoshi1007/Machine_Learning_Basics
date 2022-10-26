import pandas as pd
import mlxtend
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules

dataset = [['TFIOS', '5 feet apart', 'Walk to remember'],
['TFIOS', 'Walk to remember'],
['Divergent', 'Harry Potter','Walk to remember'],
['TFIOS', '5 feet apart'],
['Divergent','Harry Potter']]
te = TransactionEncoder()
te_array = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_array, columns=te.columns_)

from mlxtend.frequent_patterns import fpgrowth
frequent_itemsets_fp=fpgrowth(df, min_support=0.2, use_colnames=True)
rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.8)
print(df)
print("\n")
print(rules_fp)
print("\n")
print(frequent_itemsets_fp)