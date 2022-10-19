# Decision Tree using CART Implementation  

import numpy as np

data_header = ["outlook", "temperature", "humidity", "windy", "play"]
data = np.array([["sunny", "hot", "high", "false", "no"],
                 ["sunny", "hot", "high", "true", "no"],
                 ["overcast", "hot", "high", "false", "yes"],
                 ["rainy", "mild", "high", "false", "yes"],
                 ["rainy", "cool", "normal", "false", "yes"],
                 ["rainy", "cool", "normal", "true", "no"],
                 ["overcast", "cool", "normal", "true", "yes"],
                 ["sunny", "mild", "high", "false", "no"],
                 ["sunny", "cool", "normal", "false", "yes"],
                 ["rainy", "mild", "normal", "false", "yes"],
                 ["sunny", "mild", "normal", "true", "yes"],
                 ["overcast", "mild", "high", "true", "yes"],
                 ["overcast", "hot", "normal", "false", "yes"],
                 ["rainy", "mild", "high", "true", "no"]])
features = ["sunny", "overcast", "rainy", "hot", "mild", "cool", "high", "normal", "false", "true"]

#calculate the frequency distribution
frequency = {}
for i in features:
    counter = 0
    counter_two = 0
    for j in data:
        for k in j:
            if k == i and j[4] == "yes":
                counter += 1
                frequency[k] = counter, counter_two
            elif k == i and j[4] == "no":
                counter_two += 1
                frequency[k] = counter, counter_two

#entropy
entropy = {}
def Entropy(yes, no):
    t = yes+no
    return -(yes/t * np.log2(yes/t))-(no/t * np.log2(no/t))
for m in frequency:
    entropy[m] = Entropy(frequency[m][0], frequency[m][1])
    
#overall entropy
all_entropy = 0
yes_count = 0
column = data.shape[0]
for n in data:
    if "yes" in n:
        yes_count += 1
no_count = column-yes_count

all_entropy = -(yes_count/column * np.log2(yes_count/column))-(no_count/column * np.log2(no_count/column))   

#information gain
def InformationGain(avrg, *args):
    information = 0
    for i in range(len(avrg)):
        information += avrg[i]/14 * args[i]
    return all_entropy-information

#oulook information gain
avrg = (5, 4, 5)
IG = InformationGain(avrg, entropy["sunny"], 0, entropy["rainy"])
print("\nOulook Information Gain =", IG)

#temperature information gain
avrg = (4, 6, 4)
IG = InformationGain(avrg, entropy["hot"], entropy["mild"], entropy["cool"])
print("Temperature Information Gain =", IG)

#humidity information gain
avrg = (7, 7)
IG = InformationGain(avrg, entropy["high"], entropy["normal"])
print("Humidity Information Gain =", IG)

#windy information gain
avrg = (8, 6)
IG = InformationGain(avrg, entropy["false"], entropy["true"])
print("Windy Information Gain =", IG)
