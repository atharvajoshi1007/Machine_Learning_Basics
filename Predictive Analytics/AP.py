
"""
Name : Sameer Manik Patil
Div : D
Batch : D3
Roll no : 243
PRN : 0220200017
Email : sameer.patil@mitaoe.ac.in

"""

"""
ALGORITHM --------->
create dictionary
minsup
make a set
run the combinations for x 
    find sup(x)
    if sup(x) >= minsup: 
        run combinations for x U y
            find sup(y)
            conf x / y
            add conf to the frequent itemsem 
"""

import timeit
import pandas as pd
from itertools import combinations

"""
Transactions = {
                    'Abhijeet' : {"Tea", "Parle-G"},
                    'Sameer' : {"Tea"},
                    'Aditya' : {"Coffee"},
                    'Hitesh' : {"Coffee", "Parle-G"},
                    'Yogesh' : {"Tea", "Cream Roll", "Vada-pav"},
                    'Sushil' : {"Cream Roll", "Tea", "Vada-pav"},
                    'Amey' : {"Tea", "Parle-G"},
                    'Tejas' : {"Coffee", "Cream Roll", "Vada-pav"},
                    'Sunil' : {"Tea", "Vada-pav"},
                    'Pranil' : {"Tea"}
                }
"""

def dataPreProcessing(Transactions):
    data = pd.read_csv("D:\Programming\Python\Data Science\Predictive Analysis\Datasets\MBA_Chaiwala.csv")
    L1 = []; L2 = []
    for i in data["Customers"].values.tolist() : L1 += [[i]] 
    for i in data["Items"].values : L2 += [[i]] 
    for i in range(len(L2)) : L2[i] = [k for k in L2[i][0].split(',')]
    for i in range(len(data)) : Transactions.update({L1[i][0]:set(L2[i])})
    return Transactions


MINIMUM_SUPPORT = 10
MINIMUM_CONFIDENCE = 10
Transactions = {}
Itemset = set()

Transactions = dataPreProcessing(Transactions)

for items in Transactions.values(): 
    Itemset = Itemset.union(items)      # Extract total unique items from the transactions
L = len(Transactions)

print("\nMinimum Support Considered =",MINIMUM_SUPPORT, f"({MINIMUM_SUPPORT/L})", f"\nMinimum Confidence = {MINIMUM_CONFIDENCE} %", "\nTotal items =", len(Itemset), "\nItems =",Itemset)

Frequent_X = {}

start = timeit.default_timer()

for i in range(1,len(Itemset)+1):
    
    X_list = list(combinations(Itemset, i))     # generate candidates for the specified number of combinations
    
    for X in X_list:
        X_support = 0
        
        for I in Transactions.values():
            if set(X).issubset(I):      # First find count of X_itemset from all Transactions
                X_support += 1

        if X_support < MINIMUM_SUPPORT:
            continue   # we only consider itemsets that satisfy our Minimun support

        for j in range(1,len(Itemset)+1):
            
            Y_support = 0
            Y_list = list(combinations(Itemset, j))

            for Y in Y_list:

                for T in Transactions.values():
                    if T == set(Y):   # count of Y in transaction set
                        Y_support += 1

                if Y == X or len(set(Y).intersection(set(X)))!=0:  continue     # do not repeat for same associations

                X_union_Y_support = 0

                for T in Transactions.values():
                    if T == set(Y).union(set(X)):   # count of X union Y in transaction set
                        X_union_Y_support += 1

                if X_union_Y_support >= MINIMUM_SUPPORT and (X_union_Y_support/X_support*100)>MINIMUM_CONFIDENCE:
                    
                    confidence = round(X_union_Y_support/X_support,4)
                    support_X = round(X_support/L,4)
                    support_Y = round(Y_support/L, 4)
                    support_XuY = round(X_union_Y_support/L,4)
                    lift = round(confidence / (support_Y+0.01), 4)              # Lift measures how many times more often X and Y occur together than expected if they where statistically independent.
                    conviction = round((1-(support_Y)) / (1-confidence), 2)     # compares the probability that X appears without Y if they were dependent with the actual frequency of the appearance of X without Y
                    leverage = round(support_XuY - support_X * support_Y, 2)    # measures the difference of X and Y appearing together in the data set and what would be expected if X and Y where statistically dependent
                    Frequent_X.update({tuple(X):[set(Y), support_X, support_XuY, round(confidence*100,2), round(lift,2), conviction, leverage]})


# Now Lets print the Frequent Itemsets that we have filtered
print("\n")
for key, value in Frequent_X.items():
    print(list(set(key)), " -----> ", value[0], "     Sup(XuY) = ", value[2], "    Sup(X) = ", value[1], " confidence =", value[3],"%", "    Lift = ", value[4], "     Conviction = ", conviction, "   Leverage = ", leverage)

stop = timeit.default_timer()           
print('\n\nExecution Time: ', round(stop - start, 5),"sec\n\n")   # Execution time for the algorithm



