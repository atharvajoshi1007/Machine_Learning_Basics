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

#-------------------------------- Import Libraries -------------------------------------------------------------------------------

import timeit
import pandas as pd
from tabulate import tabulate
from itertools import combinations

#-------------------------------- Data Processing and Sampling -----------------------------------------------------------------------------

def dataPreProcessing(Transactions):
    data = pd.read_csv("D:\Programming\Python\Data Science\Predictive Analysis\Datasets\MBA_Chaiwala.csv")
    Sample_Size = float(input("Enter the Sample Size Percentage : "))   # Percentage in range of 0 to 1
    data = pd.DataFrame(data).sample(frac=Sample_Size)      # Take sample as a percentage of total itemset
    print(tabulate(data, tablefmt='psql'))
    L1 = []; L2 = []
    for i in data["Customers"].values.tolist() : L1 += [[i]] 
    for i in data["Items"].values : L2 += [[i]] 
    for i in range(len(L2)) : L2[i] = [k for k in L2[i][0].split(',')]
    for i in range(len(data)) : Transactions.update({L1[i][0]:set(L2[i])})
    return Transactions

#-------------------------------- Global Declaration -----------------------------------------------------------------------------

MINIMUM_SUPPORT = int(input("Enter the Minimum suppiort count : "))
MINIMUM_CONFIDENCE = int(input("Enter the Minimum Confidence : "))
Transactions = {}
Itemset = set()

Transactions = dataPreProcessing(Transactions)

for items in Transactions.values(): 
    Itemset = Itemset.union(items)      # Extract total unique items from the transactions
L = len(Transactions)

print("\nMinimum Support Considered =",MINIMUM_SUPPORT, f"({MINIMUM_SUPPORT/L})", f"\nMinimum Confidence = {MINIMUM_CONFIDENCE} %", "\nTotal items =", len(Itemset), "\n Unique Items =",Itemset)

Frequent_X = pd.DataFrame(columns=['Buy', 'Recommend', 'Support(X)','Support(XuY)','Confidence (%)','Lift','Conviction','Leverage'])

#-------------------------------- Apriori -----------------------------------------------------------------------------

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
                    try:
                        conviction = round((1-(support_Y)) / (1-confidence), 2)     # compares the probability that X appears without Y if they were dependent with the actual frequency of the appearance of X without Y
                    except:
                        conviction = "infinity"
                    leverage = round(support_XuY - support_X * support_Y, 2)    # measures the difference of X and Y appearing together in the data set and what would be expected if X and Y where statistically dependent
                    #Frequent_X.update({tuple(X):[set(Y), support_X, support_XuY, round(confidence*100,2), round(lift,2), conviction, leverage]})

                    Frequent_X = Frequent_X.append({
                                                    'Buy':set(X),
                                                    'Recommend':set(Y),
                                                    'Support(X)':support_X,
                                                    'Support(XuY)':support_XuY,
                                                    'Confidence (%)':round(confidence*100,2),
                                                    'Lift':round(lift,2),
                                                    'Conviction':conviction,
                                                    'Leverage':leverage,
                                                    },ignore_index = True)


#------------------------------------------ Display Result ------------------------------------------------------------------------------------------------------------------
# Now Lets print the Frequent Itemsets that we have filtered

print("\n")
print(tabulate(Frequent_X, headers='keys', tablefmt='psql'))
print('\n\n')

stop = timeit.default_timer()           
print('\n\nExecution Time: ', round(stop - start, 5),"sec\n\n")   # Execution time for the algorithm

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
