
# --------------------------- Import -------------------------------------------------------

from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# ---------------------------- Read Dataset and Visualize ------------------------------------------------------
df = pd.read_csv('Datasets\Income.csv')
print(df.head())

plt.scatter(df.Age,df['Income($)'],label='income')
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()
plt.show()

# ------------------------------ K-means ----------------------------------------------------
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age','Income($)']])
print("Y-Predicted Values -------> ", y_predicted)

df['cluster']=y_predicted
df.head()
print(km.cluster_centers_)

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.legend()
plt.show()

# --------------------------- Feature Scaling -------------------------------------------
scaler = MinMaxScaler()
scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])
scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])

print(df.head())

plt.scatter(df.Age,df['Income($)'],label='income')
plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.legend()
plt.show()

# ---------------------------- K-means --------------------------------------------------

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age','Income($)']])
print("Y-Predicted -------> ", y_predicted)

df['cluster']=y_predicted
print(df.head())

print(km.cluster_centers_)

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.Age,df1['Income($)'],color='green',label='Income - 1')
plt.scatter(df2.Age,df2['Income($)'],color='red',label='Income - 2')
plt.scatter(df3.Age,df3['Income($)'],color='black',label='Income - 3')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.legend()
plt.show()

# ---------------------------- Elbow method --------------------------------------------------

sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age','Income($)']])
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()


