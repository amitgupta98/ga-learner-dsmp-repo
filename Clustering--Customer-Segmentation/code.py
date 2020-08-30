# --------------
# import packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 



# Load Offers
offers=pd.read_excel(path, sheet_name=0)
print(offers.shape)
# Load Transactions
transactions=pd.read_excel(path, sheet_name=1)
transactions['n']=1
# print(transactions['n'])
# Merge dataframes
# frames=[Offers,transactions]
# df=pd.concat(frames)
df=pd.merge(offers,transactions)
# print(df.shape)

# Look at the first 5 rows



# --------------
# Code starts here

# create pivot table
matrix=df.pivot_table(index='Customer Last Name', columns='Offer #', values='n')
matrix
# replace missing values with 0
matrix.fillna(0,inplace=True)

# reindex pivot table
matrix.reset_index(inplace=True)

# display first 5 rows
matrix.head()

# Code ends here


# --------------
# import packages
from sklearn.cluster import KMeans

# Code starts here

# initialize KMeans object
cluster = KMeans(
    n_clusters=5, init='k-means++',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)

# create 'cluster' column
matrix['cluster']=cluster.fit_predict(matrix[matrix.columns[1:]])
matrix
# Code ends here


# --------------
# import packages
from sklearn.decomposition import PCA
# import matplot

# Code starts here

# initialize pca object with 2 components
pca=PCA(n_components=2, random_state=0)

# create 'x' and 'y' columns donoting observation locations in decomposed form
matrix['x']=pca.fit_transform(matrix[matrix.columns[1:]])[:,0]
matrix['y']=pca.fit_transform(matrix[matrix.columns[1:]])[:,1]
# dataframe to visualize clusters by customer names
clusters=matrix.iloc[[0,33,34,35]]

# visualize clusters
clusters.plot.scatter(x='x', y='y', c='cluster', colormap='viridis')

# Code ends here


# --------------
# Code starts here

# merge 'clusters' and 'transactions'
# data=pd.merge(clusters,transactions)
# data=pd.merge(clusters,transactions,on='Customer Last Name')

# # merge `data` and `offers`
# data=pd.merge(offers,data)
# # initialzie empty dictionary
# champagne={}

# # iterate over every cluster
# for i in range(0,5):
#     # observation falls in that cluster
#     # new_df=pd.DataFrame()
#     df=data[data['cluster']==i]
#     # sort cluster according to type of 'Varietal'
#     # counts=new_df.value_counts(ascending=False)
#     counts = df['Varietal'].value_counts(ascending=False)
#     # check if 'Champagne' is ordered mostly
#     # if (counts[0]=='Champagne'):
#     if counts.index[0] == 'Champagne':
#         # add it to 'champagne'
#         # champagne.append(counts[0])
#         champagne.update({i: counts[0]})

# # get cluster with maximum orders of 'Champagne' 
# # cluster_champagne= champagne.max()
# cluster_champagne = max(champagne, key=champagne.get)

# # print out cluster number
# # print(cluster_champagne)
# print(data['cluster'])


# Code starts here

# merge 'clusters' and 'transactions'
data=pd.merge(clusters,transactions,on='Customer Last Name')

# merge `data` and `offers`
data=pd.merge(offers,data)
# initialzie empty dictionary
champagne={}

# iterate over every cluster
for i in range(0,5):
    # observation falls in that cluster
    df=data[data['cluster']==i]
    # sort cluster according to type of 'Varietal'
    counts = df['Varietal'].value_counts(ascending=False)
    # check if 'Champagne' is ordered mostly
    if counts.index[0] == 'Champagne':
        # add it to 'champagne'
        champagne.update({i: counts[0]})

# get cluster with maximum orders of 'Champagne' 
cluster_champagne = max(champagne, key=champagne.get)

# print out cluster number
print(data['cluster'])




# --------------
# Code starts here

# empty dictionary
discount={}

# iterate over cluster numbers
for i in range(0,5):
    # dataframe for every cluster
    new_df=data[data['cluster']==i]
    # average discount for cluster
    average=sum(new_df['Discount (%)'])/len(new_df)
    # adding cluster number as key and average discount as value 
    discount.update({i:average})

# cluster with maximum average discount
cluster_discount = max(discount,key=discount.get)

# Code ends here


