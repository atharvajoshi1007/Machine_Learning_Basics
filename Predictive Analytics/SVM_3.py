
# --------------------------------------- Importing Libraries -------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ------------------------------------Import datasets and initialization ---------------------------
dataset = pd.read_csv('Datasets/Iris.csv')
dataset.head()
X = dataset.iloc[:,:4].values
y = dataset['Label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 82)
print(X_train,X_test,y_train,y_test)

# ------------------------------------------ Data transformation --------------------------------------------
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ----------------------------------------SVM Classifier ------------------------------------------------------
svcclassifier = SVC(kernel = 'linear', random_state = 0)
svcclassifier.fit(X_train, y_train)

y_pred = svcclassifier.predict(X_test)
print(y_pred)

y_compare = np.vstack((y_test,y_pred)).T
#actual value on the left side and predicted value on the right hand side
#printing the top 5 values
print(y_compare[:5,:])

# -------------------------------------------- Consfusion Matrix -------------------------------------------
cm = confusion_matrix(y_test, y_pred)
print(cm)

a = cm.shape
corrPred = 0
falsePred = 0

# ---------------------------------- Accuracy ----------------------------------------------------------
for row in range(a[0]):
    for c in range(a[1]):
        if row == c:
            corrPred +=cm[row,c]
        else:
            falsePred += cm[row,c]
print('Correct predictions: ', corrPred)
print('False predictions', falsePred)
kernelLinearAccuracy = corrPred/(cm.sum())
print ('Accuracy of the SVC Clasification is: ', corrPred/(cm.sum()))

# -------------------------------------------------------------------------------------------------

Testing_data = np.array([5.3,3,4,1.5], [ 5.5,2.2,4.8,1.82 ], [3.3,2.5,1.6,0.9])

