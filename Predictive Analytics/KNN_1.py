

""" ------* KNN Classification Algorithm*---------

1 Take any dataset where classification can be done
2 If there are non numeric values, then first convert it into numeric
3 Assign the weights for each column
4 Decide the K-value
    #larger k value means smother curves of separation resulting in less complex models. 
    # Whereas, smaller k value tends to overfit the data and resulting in complex models.
5 find the Eucledian distance among the datapoints
6 Classify points based on the Eucledian distance and weights
7 for any new value 
    8 use the stored value
    8 calculate the distance of the new point
    9 Make predictions based on the distance and assigned weights
10 Print Accuracy and error rate of the predictions
"""
# ----------------------- Imports -----------------------------------------------------

from msilib.schema import Error
from tkinter import E
from unittest import expectedFailure
import pandas as pd
import numpy as np
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------- Methods -------------------------------------------------------

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    total_distance = 0.0
    #print(row1,row2)
    for i in range(len(row1)-1):
        distance = (row1[i] - row2[i])**2
        if i == 1:
            distance **= 2  # Add Weightage to the distance for CGPA
        total_distance+=distance
    return sqrt(total_distance)
 
# Locate the most similar neighbors
def get_neighbors(dataset, test_row, num_neighbors):
    distances = list()
    
    for train_row in dataset.values:
        #print("Test Row ---->", test_row)
        dist = euclidean_distance(test_row, train_row[:-1])
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

# Predict the class of the test row
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

# ------------------ Declarations ------------------------------------------------------

# class ------> 9.5-10 (0) | 9.0-9.49 (1) | 8 - 8.99 (2) | 7 - 8 (3) | <7 (0)

dataset = pd.read_csv("Datasets\placements2.csv")

# converting String output classes into numeric class values
dataset["eligible"].replace({"Eligible": 1, "Not Eligible":0}, inplace=True)   
dataset["internships"].replace({"Yes": 1, "No":0}, inplace=True)   

K = int(input("\n\nEnter the K-value : "))
Error = 0
Test_Row = [9.1,0,0]

neighbors = get_neighbors(dataset, Test_Row[:-1], K)
print("\n\nNearest Neighbours ----------->\n")
for neighbor in neighbors:
	print(neighbor)

# -------------------------- Classification -------------------------------------------------------------

predictions = []
for row in dataset.values:
    prediction = predict_classification(dataset, row, K)
    predictions.append(prediction)
    #print('Expected %d, Got %d.' % (row[-1], prediction))
    if row[-1]!=prediction:
        Error += 1

print("\nAccuraccy : ", (len(dataset)-Error)*100/len(dataset))
print("Error : ",Error*100/len(dataset),"\n\n")

dataset['predictions'] = predictions
dataset['misclassified'] = [0]*len(dataset)
for i in range(len(dataset)):
    if dataset.loc[i,'predictions']  != dataset.loc[i,'eligible']:
        dataset.loc[i,'misclassified'] = 1

# ---------------------- Plots ----------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)    
sns.scatterplot(ax=axes[0],data=dataset, x='cgpa', y='certifications', hue='eligible')
axes[0].set_title("Expected eligibilty")
axes[0].legend(loc="upper left", labels=['Eligible for Campus Placement', 'Not eligible for Campus Placements'])
sns.scatterplot(ax=axes[1],data=dataset, x='cgpa', y='certifications', hue='predictions')
axes[1].set_title("Predicted eligibilty")
axes[1].legend(loc="upper left", labels=['Eligible for Campus Placement', 'Not eligible for Campus Placements'])
sns.scatterplot(ax=axes[2],data=dataset, x='cgpa', y='certifications', hue='misclassified')
axes[2].set_title("Mis-classified points\n1---->Miss classified\n0----->Correctly Classified")
axes[2].legend(loc="upper left")

#sns.stripplot(x="Species", y="PetalLengthCm", data=df, jitter=True, edgecolor="gray")

plt.show()

# ---------------------------------------------------------------------------------------
