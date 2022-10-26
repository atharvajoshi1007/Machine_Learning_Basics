
"""
# Algorithm for Multilevel Assocaition mining 

format the dataset such data we have category column

1. Read product heirarchy from user
    from generic categories upto subcategories

2. Specify minsup for each level

3. Run for itemsets at every level:
        Run for all combination of itemsets at the level
            Count support of each itemset in all transactions (one for each transaction)
            Prune candidates having sup(x) < minsup

4. Combinations that qualify minsup at respective levels are taken for the association
"""

# ------------------------------Import--------------------------------------------
#from math import prod
from re import template
from threading import local
from tabulate import tabulate
import timeit
from unicodedata import category
import pandas as pd
from itertools import combinations

# ------------------------------ Methods -------------------------------
# Data Preprocessing
def dataPreProcessing(Transactions):
    data = pd.read_csv("D:\Programming\Python\Data Science\Predictive Analysis\Datasets\shop.csv")
    L1 = []; L2 = []
    for i in data["Customers"].values.tolist() : L1 += [[i]] 
    for i in data["Items"].values : L2 += [[i]] 
    for i in range(len(L2)) : L2[i] = [k for k in L2[i][0].split(',')]
    for i in range(len(data)) : Transactions.update({L1[i][0]:set(L2[i])})
    return Transactions

# get level-local itemset
def getLocalItemset(Itemset):
    local_itemset = set()
    for I in Itemset:
        temp_items = I.split('|')       # Here we examine each item for its respective level by splitting through '|'
        if len(temp_items) >= level:
            local_itemset.add(temp_items[level-1])       
    return local_itemset

# Print Support Count
def printSupportCount(Local_Itemset):
    DF = pd.DataFrame()
    for key,value in Local_Itemset.items():
        DF = DF.append({'Items':key, 'Support_Count':value}, ignore_index=True)
    DF.index+=1
    print(tabulate(DF.sort_values('Support_Count', ascending=False), headers='keys', tablefmt='psql'))


# print parent reference
def getRootPath(X, Itemset, level):
    temp = []   
    for x in X:
        for I in Itemset:
            if x in I:
                i = '|'.join(I.split('|')[:level])
                temp.append(i)      # append absolute path of item
                break
    return tuple(temp)

# get item Support count
def getSupportCount(X, Transactions):
    X_support = 0
    for T in Transactions.values():
        x_count = 0
        for x in X:
            for t in T:
                if x in t:
                    x_count += 1
                    break    
        if x_count==len(X):
            X_support += 1
    return X_support

# get Maximum possible levels
def getMaxLevels(Itemset):
    levels = 0
    for l in Itemset:
        if len(l.split('|')) > levels: levels = len(l.split('|'))
    return levels
# ------------------------------Prerequisites--------------------------------------

MINIMUM_CONFIDENCE = 2
Frequent_X = {}
Transactions = {}
Itemset = set()
Transactions = dataPreProcessing(Transactions)

for items in Transactions.values(): 
    Itemset = Itemset.union(items)      # Extract total unique items from the transactions
L = len(Transactions)                   # total unique items
MAX_LEVELS = getMaxLevels(Itemset)
MINIMUM_SUPPORT = MAX_LEVELS + 1        # set the minimum support for highest level as MAX Level
print("\nTotal Transactions: ",L)
print("\nTotal items: ",len(Itemset))
print('\nUnique Items available in the dataset: \n', Itemset, "\n")

#-----------------------------------Apriori----------------------------------------------------#

def Apriori(Unique_Items, MINSUP, Transactions):
    Frequent_DF = pd.DataFrame(columns=['Buy', 'Recommend', 'Support(X)','Support(XuY)','Confidence','Lift','Conviction','Leverage'])

    for i in range(1,len(Unique_Items)):
        X_support = 0
        X_list = list(combinations(Unique_Items, i))

        for X in X_list:
            #if isinstance(X, str): getSupportCount(X.split(','),Transactions)
            #else: X_support = getSupportCount(X,Transactions)
            X_support = getSupportCount(X,Transactions)

            if X_support >= MINSUP:
            
                for j in range(1,len(Unique_Items)):
                    Y_support = 0
                    Updated_Unique_Items = Unique_Items.copy() 
                    for x in X:
                        Updated_Unique_Items.pop(x)
                    
                    Y_list = list(combinations(Updated_Unique_Items , j))
                    
                    for Y in Y_list:              # go through All possible combinations
                        #if Y == X or X in Y or Y in X:  continue     # do not repeat for same associations

                        #if isinstance(Y, str): getSupportCount(Y.split(','),Transactions)
                        #else: Y_support = getSupportCount(Y,Transactions)
                        Y_support = getSupportCount(Y,Transactions) ##
                        X_union_Y_support = 0
                        XuY_List = set()

                        #if isinstance(X, str) or len(X)==1: XuY_List.add(X[0]) 
                        #else:  XuY_List = XuY_List.union(set(X)) 
                        XuY_List = XuY_List.union(set(X)) ##
                        #if isinstance(Y, str) or len(Y)==1: XuY_List.add(Y[0])
                        #else:  XuY_List = XuY_List.union(Y)
                        XuY_List = XuY_List.union(Y) ## 
                        if len(XuY_List)>0:
                            X_union_Y_support = getSupportCount(XuY_List,Transactions)

                        if X_union_Y_support >= MINSUP:
                            
                            confidence = round(X_union_Y_support/X_Support,4)
                            support_X = round(X_Support/L,4)
                            support_Y = round(Y_support/L, 4)
                            support_XuY = round(X_union_Y_support/L,4)
                            lift = round(confidence / (support_Y), 4)              # Lift measures how many times more often X and Y occur together than expected if they where statistically independent. (correlation)
                            try:
                                conviction = round((1-(support_Y)) / (1-confidence), 2)     # compares the probability that X appears without Y if they were dependent with the actual frequency of the appearance of X without Y
                            except:
                                conviction = 999
                            leverage = round(support_XuY - support_X * support_Y, 2)    # measures the difference of X and Y appearing together in the data set and what would be expected if X and Y where statistically dependent
                            Frequent_DF = Frequent_DF.append({
                                                    'Buy':set(X),
                                                    'Recommend':set(Y),
                                                    'Support(X)':support_X,
                                                    'Support(XuY)':support_XuY,
                                                    'Confidence':round(confidence*100,2),
                                                    'Lift':round(lift,2),
                                                    'Conviction':conviction,
                                                    'Leverage':leverage,
                                                    },ignore_index = True)

    print("\n")
    print(Frequent_DF.to_string(index=False))
    print('\n\n')


# ------------------------------Create Hierarchy--------------------------------------------


for level in range(1, MAX_LEVELS+1):    # go through each level
    
    MINIMUM_SUPPORT -= 1
    
    local_itemset = getLocalItemset(Itemset)                # level wise local itemset
    
    print("\n\x1b[6;30;42m Items at Level ",level,":",local_itemset, "     Minimum Support:",MINIMUM_SUPPORT,"  "); print('\x1b[0m')
    
    for i in range(1,len(local_itemset)+1):     
        
        Local_X_List = list(combinations(local_itemset, i)) # generate candidates for the specified i combinations
        Support_List = []
        Local_Unique_Items = {}
        Local_Frequent_itemset = {}

        for X in Local_X_List:
            
            X_Support = getSupportCount(X, Transactions)    # get support of individual itemset
            if X_Support < MINIMUM_SUPPORT:                 # pruning
                continue                                    # we only consider itemsets that satisfy our Minimun support
            
            if level>1: X = getRootPath(list(X), Itemset, level)   # print along with parent reference
            
            if len(X)==1 : Local_Unique_Items.update({X[0]:X_Support})         # get Unique items at each level
            
            Local_Frequent_itemset.update({X:X_Support})                # save the local frequent itemset

        if len(Local_Frequent_itemset)>0:                   # consider only non-empty frequent set levels
            print("\n\x1b[7;30;43m Support count for Level - ", level, " and combination of length - ", i); print('\x1b[0m')
            printSupportCount(Local_Frequent_itemset)
            
            if len(Local_Unique_Items) > 0:
                Apriori(Local_Unique_Items,level,Transactions)
                print(Local_Unique_Items)
