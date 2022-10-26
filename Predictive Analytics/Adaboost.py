# Adaboost using Libraries

# ---------------------------------- Importing Essential Libraries -----------------------------------------
from statistics import mode
import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns

# ---------------------------------- Dataset Initialization -----------------------------------------

dataset = pd.read_csv("Datasets\Cricket.csv")

print("\n\nDataset Shape -----> ", dataset.shape)
print("\n\nDataset.head() -------> \n\n", dataset.head())
print("\n\nDataset.describe() -------> \n\n", dataset.describe())

# ---------------------------------- Data preprocessing --------------------------------------------
# Converting nominal values into continuous values 
dataset["Team Wins"].replace({"Yes": 1, "No":0}, inplace=True)   
dataset["Venue"].replace({"India": 1, "Overseas":0}, inplace=True)   
dataset["Format"].replace({"Test": 0, "T20":1, "ODI":2}, inplace=True)   
dataset["Opponent"].replace({"Australia": 0, "England":1, "West-Indies":2, "South-Africa":3, "New-Zealand":4, "Pakistan":5, "Sri-Lanka":6}, inplace=True)   

data_hash = {"Venue":{1:"India", 0:"Overseas"}, "Format":{0:"Test", 1:"T20", 2:"ODI"}, "India Wins":{1:"Yes", 0:"No"},
             "Opponent":{0:"Australia", 1:"England", 2:"West-Indies", 3:"South-Africa", 4:"New-Zealand", 5:"Pakistan", 6:"Sri-Lanka"}
            }

X_Data =[]
Y_Data = []
  
# Iterate over each row to fetch row data in a list 
for index, rows in dataset.iterrows():
    # Create list for the current row
    x_list = [rows.Opponent, rows.Venue, rows.Format]
    # append the list to the final list
    X_Data.append(x_list)
    Y_Data.append(rows["Team Wins"])
  
# ---------------------------------- Model Training and Prediction --------------------------------------------

X = X_Data 
Y = Y_Data 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1) # 70% training and 30% test

abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)
# Train Adaboost Classifer
model = abc.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = model.predict(X_test)

print("\n\nModel ------>", model)

print(f"\n\nAccuracy : {round(metrics.accuracy_score(Y_test, Y_pred)*100, 2)}%")

# Testing our prediction with an incident example
Validation_list = [2, 1, 1]
model = abc.fit(X, Y)
# prediction for the incident example
Y_pred = model.predict([Validation_list])

print("\n\nOpponent :", data_hash["Opponent"][Validation_list[0]],"  | Venue :", data_hash["Venue"][Validation_list[1]], "  | Format :", data_hash["Format"][Validation_list[2]], " ------->  India Wins? ====> ", data_hash["India Wins"][Y_pred[0]],"\n\n")

# -------------------------- Visualization of Accuracy and n_estimators -------------------------------------
for n,i in zip(range(1,100, 5), range(1,20)):
    abc = AdaBoostClassifier(n_estimators=n, learning_rate=i/20)
    # Train Adaboost Classifer
    model = abc.fit(X_train, Y_train)
    plt.scatter(n, abc.score(X_test, Y_test))
plt.xlabel("Estimator Decision Trees")
plt.ylabel("Accuracy Score")
plt.show()

