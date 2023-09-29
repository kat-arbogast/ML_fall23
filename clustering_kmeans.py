'''
Get ready to cluster some fire data

Code is referenced from Dr. Gates' tutorial on KMeans By Hand in Python found at https://gatesboltonanalytics.com/?page_id=924 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from IPython.display import clear_output
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets



#------------------------------------------

# Permanently changes the pandas settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

## This is MY path - you need to update this to 
## YOUR path :)
## You should update this to read in any 
## labeled record dataset that is numeric. 

path="C:/Users/profa/Desktop/UCB/ML CSCI 5622/Data/SummerStudentAdmissions_CLEAN_numeric.csv"

DF=pd.read_csv(path)
#print(DF)

## Remove and save the label
## When you use YOUR data - you 
## will update this so that
## you remove the label from
## your data. 

DFLabel=DF["Decision"]
#print(DFLabel)

DF=DF.drop(["Decision"], axis=1)
#print( DF)

k=3
## Function that creates random centroids
def RandomCentroidInit (DF, k):
    MyCentroids=[]
    for i in range(k):
        nextcentroid=DF.apply(lambda x: float(x.sample()))
        MyCentroids.append(nextcentroid)
    return pd.concat(MyCentroids, axis=1)

MyCentroids=RandomCentroidInit(DF, k)
print(MyCentroids)

## Function that finds distances between all points and all centroids
## Label each point with a centroid (starting at 0)
def Label_Data(DF, MyCentroids):
    dist=MyCentroids.apply(lambda x: np.sqrt(((DF - x)**2).sum(axis=1)))
    labels=dist.idxmin(axis=1)
    return labels

cluster_labels=Label_Data(DF,MyCentroids)
#print(cluster_labels)
## How many points are in each label/cluster right now
print(cluster_labels.value_counts())

def Updated_Centroids(DF, cluster_labels, k):
    Cluster_Means=DF.groupby(cluster_labels).apply(lambda x: x.mean()).T
    return Cluster_Means

MyCentroids=Updated_Centroids(DF, cluster_labels, k)
print(MyCentroids)

##################### PCA if your data is ....




def ClusterPlot(DF, cluster_labels, MyCentroids, iteration):
    MyPCA=PCA(n_components=2)
    Data2D = MyPCA.fit_transform(DF)
    Centroids2D=MyPCA.transform(MyCentroids.T)
    clear_output(wait=True)
    plt.title("Clustering")
    plt.scatter(x=Data2D[:,0], y =Data2D[:,1],  c=cluster_labels )
    plt.scatter(x =Centroids2D[:,0], y= Centroids2D[:,1],s=200, alpha=0.5)
    plt.show()
    

## Iterate
NumInterations = 20
iteration=1

while iteration < NumInterations:
    print("Iteration: ", iteration)
    cluster_labels=Label_Data(DF, MyCentroids)
    MyCentroids=Updated_Centroids(DF, cluster_labels, k)
    print("Centroids:\n", MyCentroids)
    ClusterPlot(DF, cluster_labels, MyCentroids, iteration) 
    iteration = iteration + 1

