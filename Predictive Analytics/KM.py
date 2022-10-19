
"""
Algorithm:
    
    1. Initially, Consider any n-data points as centroids
    2. Calculate the object distances from the centroid
    3. Create the object clustering matrix from with respect to the centroids
    4. Calculate the new centroids
    5. Repeat the process for the new centroids until there is no change in the clustering matrix

"""
# --------------------------- Importing libraries ------------------------

from cmath import sqrt
from ctypes import pointer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------- import data ---------------------------------------------------

data = pd.read_csv("Datasets\Kmeans_testing.csv")
X1 = data["Feature_1"]
X2 = data["Feature_2"]
plt.scatter(X1,X2)
plt.xlabel("Feature_1")
plt.ylabel("Feature_2")
plt.show()

# --------------------- KMeans Method ------------------------------------------

# KMeans(Feature_, Feature_2, Centroid)

def KMeans(X1, X2, C) :
    k = 0
    L = len(X1)
    Object_Distance = np.zeros([len(C),L])
    Object_Clustering_old = np.zeros([L,len(C)])
    Object_Clustering_new = np.zeros([L,len(C)])
    Centroid_Distance = 0
    Epoch = 0
    Max_Distance = []

    while True:

        Object_Clustering_new = np.zeros([L,len(C)])

    # --------------------- Object Centroid Distance ------------------------------------------

        print(f"\n\n************************* Epoch = {Epoch} ***********************\n")
        for c in C:
            for i in range(L):
                Centroid_Distance = round(sqrt((abs(X1[i]-c[0])**2)+(abs(X2[i]-c[1])**2)).real, 2)
                Object_Distance[k][i] = Centroid_Distance
            k += 1
        k = 0
        print(Object_Distance, "\n\n")

        # --------------------- Object Clustering -------------------------------------------------

        Distance_Transpose = Object_Distance.T.copy()

        for i in range(L):
            point_distance = Distance_Transpose[:][i]
            minpos = np.where(point_distance==np.min(point_distance))[0][0]
            Object_Clustering_new[i][minpos] = 1

        Object_Clustering_T = Object_Clustering_new.T.copy()
        print(Object_Clustering_T, "\n\n", C, "\n\n")
        if np.array_equal(Object_Clustering_new, Object_Clustering_old):
            break
        Object_Clustering_old = Object_Clustering_new.copy()

        # ---------------------- New Centroids ----------------------------------------------------

        for i in range(len(C)):
            point_distance = Object_Clustering_T[:][i]
            loc = np.where(point_distance==1)
            C[i][0] = round(sum(X1[k] for k in loc[0]) / len(loc[0]), 2)
            C[i][1] = round(sum(X2[k] for k in loc[0]) / len(loc[0]), 2)

        print(C)
        Epoch +=1

    # --------------------- Centroids Visualization ------------------------------------------

    for i in range(len(C)):
        point_distance = Object_Clustering_T[i][:]
        loc = np.where(point_distance==1)
        x1 = X1.iloc[loc]
        x2 = X2.iloc[loc]
        plt.scatter(x1,x2)

    plt.xlabel("Feature_1")
    plt.ylabel("Feature_2")
    for i in range(len(C)):
        plt.scatter(C[i][0], C[i][1], marker='^')
    plt.show()

    for i in range(len(C)): # Get max distance from centroid for given K
        loc = np.where(Object_Clustering_T[i][:]==1)
        Max_Distance.append(max(Object_Distance[i][loc]))

    return Epoch, C, sum(Max_Distance)/len(Max_Distance)



# --------------------- Calling KMeans ----------------------------

C = []
K_distances = []

for i in range(1,8):
    C.append([X1[i], X2[i]])    # Initializing Centroids
    Epoch, New_Centroids, Average_Distance = KMeans(X1,X2,C)
    K_distances.append(Average_Distance)
    print("Epochs ----> ", Epoch, "\nNew Centroids ----> ", New_Centroids, "\nAverage Centroid Distance ------> ", Average_Distance)

# Elbow Method --------> 

plt.xlabel('K')
plt.ylabel('Average Centroid Distance')
plt.plot(range(1,8),K_distances,'bx-')
plt.show()

"""
# -------------------------------- Elbow Method -------------------------------------------
from sklearn.cluster import KMeans
sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(data[['Feature_1','Feature_2']])
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse,'bx-')
plt.show()
"""