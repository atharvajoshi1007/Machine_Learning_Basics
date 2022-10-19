
# ---------------------------------- Importing Essential Libraries -----------------------------------------

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------- Dataset Initialization -----------------------------------------

dataset = pd.read_csv("Datasets\Cricket.csv")

print("\n\nDataset Shape -----> ", dataset.shape)
print("\n\nDataset Columns -----> ", dataset.columns)
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

RFC=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
model = RFC.fit(X_train,Y_train)

Y_pred=RFC.predict(X_test)

print("\n\nModel ------>", model)
# Model Accuracy, how often is the classifier correct?
print(f"\n\nAccuracy : -------> {round(metrics.accuracy_score(Y_test, Y_pred)*100, 2)}%")

Validation_list = [1, 1, 1]
Y_pred = RFC.predict([Validation_list])[0]
print("\n\nOpponent :", data_hash["Opponent"][Validation_list[0]],"  | Venue :", data_hash["Venue"][Validation_list[1]], "  | Format :", data_hash["Format"][Validation_list[2]], " ------->  India Wins? ====> ", data_hash["India Wins"][Y_pred],"\n\n")

feature_imp = pd.Series(RFC.feature_importances_,index=dataset.columns[:3]).sort_values(ascending=False)
print(f"\n\nFeature Importance ------->\n{feature_imp} \n\n")

# ----------------------------------- Visualization ---------------------------

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()