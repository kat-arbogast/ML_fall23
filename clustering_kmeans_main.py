'''
Get ready to cluster some fire data!!

Code is referenced from Dr. Gates' tutorial on KMeans By Hand in Python found at https://gatesboltonanalytics.com/?page_id=924 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from IPython.display import clear_output
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn import datasets

#------------------------------------------
# Permanently changes the pandas settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
#------------------------------------------

#-------------------------------------------------------------------------
## Setting the folder path to the cleaned but not formatted data
dm_state_total_area_path = "./CleanData/dm_state_total_area_cleaned.csv"
dm_state_percent_area_path = "./CleanData/dm_state_percent_area_clean.csv"
fires_monthly_folder_path = "./CleanData/us_fires_burn_monthly.csv"
us_wildfires_2mil_path = "./CleanData/us_wildfires_2mil_cleaned.csv"
or_weather_wildfires_path = "./CleanData/or_weather_wildfires_cleaned.csv"

## Setting the filename
dm_state_total_area_filename = "dm_state_total_area_cleaned"
dm_state_percent_area_filename = "dm_state_percent_area_clean"
us_fires_burn_monthly_filename = "us_fires_burn_monthly"
us_wildfires_2mil_filename = "us_wildfires_2mil_cleaned"
or_weather_wildfires_filename = "or_weather_wildfires_cleaned"
#-------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------- 
def main():
    '''
    This function takes in the cleaned fire data, transforms it for unsupervised learning, performs kmeans clustering, and provides results
    '''
    print("\n############################################################################")
    print("\n---------- Ingesting Cleaned Fire Data ----------\n")
    # Ingest Data
    dm_state_total_area = pd.read_csv(dm_state_total_area_path)
    dm_state_percent_area = pd.read_csv(dm_state_percent_area_path)
    us_fires_burn_monthly = pd.read_csv(fires_monthly_folder_path)
    us_wildfires_2mil = pd.read_csv(us_wildfires_2mil_path)
    or_weather_wildfires = pd.read_csv(or_weather_wildfires_path)
    
    print("\n---------- Transforming Fire Data for Unsupervised Learning ----------\n")
    # Transform Drought Montior Data
    dm_state_total_area_transformed = transform_data(dm_state_total_area_filename, dm_state_total_area, ['None', 'D0', 'D1', 'D2', 'D3', 'D4', 'DSCI'])
    dm_state_percent_area_transformed = transform_data(dm_state_percent_area_filename, dm_state_percent_area, ['None', 'D0', 'D1', 'D2', 'D3', 'D4', 'DSCI'])
    us_fires_burn_monthly_transformed = transform_data(us_fires_burn_monthly_filename, us_fires_burn_monthly, ['Acres_Burned', 'Number_of_Fires', 'Acres_Burned_per_Fire'])
    us_wildfires_2mil_transformed = transform_data(us_wildfires_2mil_filename, us_wildfires_2mil, ['FIRE_SIZE', 'FireDuration_hrs'])
    or_weather_wildfires_transformed = transform_data(or_weather_wildfires_filename, or_weather_wildfires, ['tmax', 'tmin', 'tavg', 'prcp', 'EstTotalAcres', 'FireDuration_hrs'])
    
    print("\n---------- Determining Optimal Number of Clusters ----------\n")
    #------------------------------------------------------------------------------------
    ## Elbow
    ## Ran these and then looked at the graphs. The values of k are then hard coded
    # determine_k_elbow(dm_state_total_area_filename, dm_state_total_area_transformed)
    # determine_k_elbow(dm_state_percent_area_filename, dm_state_percent_area_transformed)
    # determine_k_elbow(us_fires_burn_monthly_filename, us_fires_burn_monthly_transformed)
    # determine_k_elbow(us_wildfires_2mil_filename, us_wildfires_2mil_transformed)
    # determine_k_elbow(or_weather_wildfires_filename, or_weather_wildfires_transformed)
    #------------------------------------------------------------------------------------
    
    #--------------------------------------------------------------------------------------------------
    ## Silhouette
    # determining_k_silhouette(dm_state_total_area_filename, dm_state_total_area_transformed)
    # determining_k_silhouette(dm_state_percent_area_filename, dm_state_percent_area_transformed)
    # determining_k_silhouette(us_fires_burn_monthly_filename, us_fires_burn_monthly_transformed)
    
    ## Create df sample with 50% of the rows from us_wildfires_2mil_transformed
    # print(f"Sample from us_wildfires_2mil_transformed")
    # us_wildfires_2mil_transformed_sample = us_wildfires_2mil_transformed.sample(frac=0.5)
    # us_wildfires_2mil_transformed_sample.reset_index(drop=True, inplace=True)
    
    ## Create df sample with 70% of the rows from or_weather_wildfires_transformed
    # print(f"Sample from or_weather_wildfires_transformed")
    # or_weather_wildfires_transformed_sample = or_weather_wildfires_transformed.sample(frac=0.7)
    # or_weather_wildfires_transformed_sample.reset_index(drop=True, inplace=True)
    
    # determining_k_silhouette(us_wildfires_2mil_filename, us_wildfires_2mil_transformed_sample)
    # determining_k_silhouette(or_weather_wildfires_filename, or_weather_wildfires_transformed_sample)
    #--------------------------------------------------------------------------------------------------
    
    #--------------------------------------------------------------------------------------------------
    ## GAP
    determine_k_GAP(us_wildfires_2mil_filename, us_wildfires_2mil_transformed, n_bootstraps=5)
    determine_k_GAP(or_weather_wildfires_filename, or_weather_wildfires_transformed, n_bootstraps=5)
    
    #--------------------------------------------------------------------------------------------------
    
    # print("\n---------- Performing Kmeans Clustering ----------\n")
    # by_hand_kmeans(dm_state_total_area_filename, dm_state_total_area_transformed, k=2)
    # by_hand_kmeans(dm_state_percent_area_filename, dm_state_percent_area_transformed, k=2)
    # by_hand_kmeans(us_fires_burn_monthly_filename, us_fires_burn_monthly_transformed, k=2)
    # by_hand_kmeans(us_wildfires_2mil_filename, us_wildfires_2mil_transformed, k=2)
    # by_hand_kmeans(or_weather_wildfires_filename, or_weather_wildfires_transformed, k=2)

    
    print("\n\nMADE IT OUT ALIVE!!!!")
    print("\n############################################################################\n")
        
    
#-----------------------------------------------------------------------------------------------------------------   
    

def transform_data(filename, df, cols_of_interest):
    '''
    This function prepares the dataframe for unsupervised learning
        - select the columns of interest
        - collect the label column and remove all labels
        - return the dataframe and the label
    Args:
        - dataframe (cleaned record data)
        - list of the columns of interest
    Return:
        - dataframe with no labels and only the columns of interest
        - the labels        
    '''
    print(f"--- Performing Unsupervised Learning Prep Transformation for {filename} ---")
    
    df2 = df.loc[:, cols_of_interest]
    print(f"{filename} head:\n{df2.head()}")
    
    return df2
    

def determine_k_elbow(filename, df):
    '''
    This function determines the optimal number of clusters for a dataset based on the Elbow method
    
    Args:
        - dataframe that is prepped for unsupervised learning
    '''
    print(f"--- Determining the optimal number of clusters using the Elbow Method for {filename} ---")
    
    wcss = []

    # Try k from 1 to 10 clusters
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
    
    # Plot the WCSS values for different k values
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title(f'Elbow Method for Optimal k - {filename}')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.grid(True)
    plt.savefig(f"./CreatedVisuals/kmeans/Elbow/{filename}_elbow.png")
    plt.close()
    
def determining_k_silhouette(filename, df):
    '''
    This function determines the optimal number of clusters for a dataset based on the Silhouette method
    
    Args:
        - dataframe that is prepped for unsupervised learning
    '''
    
    print(f"--- Determining the optimal number of clusters using the Silhouette Method for {filename} ---")
    
    k_values = range(2, 11)  # You can adjust the range as needed
    silhouette_scores = []

    for k in k_values:
        print(f"Trying value {k}")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(df)
        print("Made cluster labels")
        
        # Calculate the silhouette score for this k
        silhouette_avg = silhouette_score(df, cluster_labels)
        print(f"---------- For k = {k}, the average silhouette_score is: {silhouette_avg}")
        silhouette_scores.append(silhouette_avg)
        
        # Calculate the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(df, cluster_labels)
        print("calculated silhouette values\n")
        
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, silhouette_scores, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Silhouette Score')
    plt.title(f'Silhouette Method for Optimal k - {filename}')
    plt.grid(True)
    plt.savefig(f"./CreatedVisuals/kmeans/Silhouette/{filename}_silhouette.png")
    plt.close()
    

def determine_k_GAP(filename, df, n_bootstraps):
    """
    Calculate the GAP statistic to find the optimal number of clusters (k) for k-means clustering.
    
    Args:
        - data: Input data for clustering (numpy array or pandas DataFrame)
        - n_bootstraps: Number of bootstrap samples for generating reference datasets.
    """
    
    print(f"--- Determining the optimal number of clusters using the GAP Statistic Method for {filename} ---")
    
    k_values = range(2, 11)
    gap_values = []
    std_devs = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        cluster_labels = kmeans.predict(df)
        print("Made cluster labels")
        
        # Calculate the within-cluster dispersion (WCSS)
        wcss = np.sum((df - kmeans.cluster_centers_[cluster_labels]) ** 2)
        print(f"---------- For k = {k}, the wcss is: {wcss}")
        
        # Generate reference datasets for comparison
        reference_wcss = []
        for _ in range(n_bootstraps):
            random_data = np.random.rand(*df.shape)
            random_kmeans = KMeans(n_clusters=k, random_state=42)
            random_kmeans.fit(random_data)
            random_labels = random_kmeans.predict(random_data)
            print("Made random labels")
            reference_wcss.append(np.sum((random_data - random_kmeans.cluster_centers_[random_labels]) ** 2))
        
        # Calculate GAP statistic
        gap = np.log(np.mean(reference_wcss)) - np.log(wcss)
        std_dev = np.sqrt(np.mean((np.log(reference_wcss) - np.log(wcss)) ** 2))
        print(f"---------- For k = {k}, the gap is: {gap}")
        print(f"---------- For k = {k}, the std_dev is: {std_dev}")
        gap_values.append(gap)
        std_devs.append(std_dev)
        
    plt.figure(figsize=(10, 5))
    plt.errorbar(range(10), gap_values, yerr=std_devs, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('GAP Statistic')
    plt.title(f'GAP Statistic for Optimal k - {filename}')
    plt.grid(True)
    plt.savefig(f"./CreatedVisuals/kmeans/GAP/{filename}_gap_stat.png")

def by_hand_kmeans(filename, df, k):
    # Initial Centroids
    MyCentroids = RandomCentroidInit(filename, df, k)
    print(f"\nINITIAL CENTROIDS: \n{MyCentroids}\n")
    
    
    ## Iterate
    NumInterations = 20
    iteration=1

    while iteration < NumInterations:
        print("\n\nIteration: ", iteration)
        
        # Create Labels for Clusters
        cluster_labels = Label_Data(filename, df, MyCentroids)
        print(f"cluster_labels value counts:\n{cluster_labels.value_counts()}\n") ## How many points are in each label/cluster right now
        
        # Create new Centroids based on labels
        MyCentroids=Updated_Centroids(filename, df, cluster_labels, k)
        print(f"Updated Centroids:\n{MyCentroids}")
                
        iteration = iteration + 1
        
    ClusterPlot(filename, df, cluster_labels, MyCentroids) 
    
    # I want to save the last photo of all the clusters
    

def RandomCentroidInit(filename, df, k):
    '''
    This function creates random k centroids
    
    Args:
        - dataframe in the proper format
        - k (the number of kmeans clusters)
    Returns:
        - dataframe where each row is the coordinate of the k centroids
    '''
    print(f"--- Randomly picking {k} centroids for {filename} ---")
    
    MyCentroids=[]
    for i in range(k):
        nextcentroid=df.apply(lambda x: float(x.sample()))
        MyCentroids.append(nextcentroid)
        
    return pd.concat(MyCentroids, axis=1)


def Label_Data(filename, df, MyCentroids):
    '''
    This function:
        - finds distances between all points and all centroids
        - labels each point with a centroid (starting at 0)
    Args:
        - dataframe
        - centroids
    Returns:
        - labels
    '''
    print(f"--- Finding Distances and Lables for {filename} ---")
    
    dist = MyCentroids.apply(lambda x: np.sqrt(((df - x)**2).sum(axis=1)))
    labels = dist.idxmin(axis=1)
    
    return labels

def Updated_Centroids(filename, df, cluster_labels, k):
    '''
    This function updates teh centriods based on the cluster_labels collected from the previous iteration of labeling
    Args:
        - filename (this is just for clear tracking)
        - dataframe
        - cluster_labels (these are generated from the function Label_Data)
        - k (the same k as defined at the beginning of the process)
    Returns:
        - new centroids
    '''
    print(f"--- Updating Centroids for {filename} ---")
    
    Cluster_Means=df.groupby(cluster_labels).apply(lambda x: x.mean()).T
    
    print(f"TYPE OF Cluster_Means: {type(Cluster_Means)}")
    return Cluster_Means


def ClusterPlot(filename, df, cluster_labels, MyCentroids):
    MyPCA=PCA(n_components=2)
    Data2D = MyPCA.fit_transform(df)
    Centroids2D=MyPCA.transform(MyCentroids.T)
    clear_output(wait=True)
    plt.title(f"Clustering {filename}")
    plt.scatter(x=Data2D[:,0], y =Data2D[:,1],  c=cluster_labels )
    plt.scatter(x =Centroids2D[:,0], y= Centroids2D[:,1],s=200, alpha=0.5)
    plt.savefig(f"./CreatedVisuals/kmeans/{filename}_kmeans.png")
    plt.close()
    
# DO NOT REMOVE!!!
if __name__ == "__main__":
    main()