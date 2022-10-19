# -*- coding: utf-8 -*-
"""PALab_ECLAT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ssis2ZkxeqsYy9xVubePY00g6H9W-nUM
"""

#pip install pyECLAT

#pip install apyori
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

transactions = [
    ['beer', 'wine', 'cheese'],
    ['beer', 'potato chips'],
    ['eggs', 'flower', 'butter', 'cheese'],
    ['eggs', 'flower', 'butter', 'beer', 'potato chips'],
    ['wine', 'cheese'],
    ['potato chips'],
    ['eggs', 'flower', 'butter', 'wine', 'cheese'],
    ['eggs', 'flower', 'butter', 'beer', 'potato chips'],
    ['wine', 'beer'],
    ['beer', 'potato chips'],
    ['butter', 'eggs'],
    ['beer', 'potato chips'],
    ['flower', 'eggs'],
    ['beer', 'potato chips'],
    ['eggs', 'flower', 'butter', 'wine', 'cheese'],
    ['beer', 'wine', 'potato chips', 'cheese'],
    ['wine', 'cheese'],
    ['beer', 'potato chips'],
    ['wine', 'cheese'],
    ['beer', 'potato chips']
]

data = pd.DataFrame(transactions)
data

# we are looking for itemSETS
# we do not want to have any individual products returned
min_n_products = 2

# we want to set min support to 7
# but we have to express it as a percentage
min_support = 7/len(transactions)

# we have no limit on the size of association rules
# so we set it to the longest transaction
max_length = max([len(x) for x in transactions])

from pyECLAT import ECLAT

# create an instance of eclat
my_eclat = ECLAT(data=data, verbose=True)

# fit the algorithm
rule_indices, rule_supports = my_eclat.fit(min_support=min_support, min_combination=min_n_products,max_combination=max_length)

print(rule_supports)

#pip install apyori

import numpy as np # to deal with numeric data
import pandas as pd # to deal with dataframe

data=pd.read_csv("Market_Basket_Optimisation.csv",header=None)
transact_list = [] # create an empty list to store transactions
for i in range(0, 7501):
  transact_list.append([str(data.values[i,j]) for j in range(0, 20)]) # add the transactions to the above created

from apyori import apriori # import the apriori library
rules = apriori(transactions = transact_list, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2) # generate rules

# list of results coming from the apriori model
rslt = list(rules)

def inspect(rslt): # function to organize the output
    left_handSide         = [tuple(result[2][0][0])[0] for result in rslt] # get the left hand side of the rules
    right_handSide         = [tuple(result[2][0][1])[0] for result in rslt] # get the right hand side of the rules
    supports    = [result[1] for result in rslt] # get the supports
    return list(zip(left_handSide,right_handSide, supports)) # zip the above three lists together
rslt_DataFrame = pd.DataFrame(inspect(rslt), columns = ['Product 1', 'Product 2', 'Support']) # create a pandas dataframe

rslt_DataFrame.nlargest(n = 7, columns = 'Support') # printing the first 7 supports