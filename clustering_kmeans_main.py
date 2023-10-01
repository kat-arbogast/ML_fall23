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
import matplotlib.cm as cm

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
or_weather_wildfires_path = "./CleanData/or_weather_wildfires_cleaned.csv"
lightining_fires_path = "./CleanData/lightning_wildfires_clean.csv"
or_weather_wildfires_comments_vector_path = "./CleanData/or_weather_wildfires_cause_comments_vectorized.csv"
or_weather_wildfires_specific_vector_path = "./CleanData/or_weather_wildfires_specific_cause_vectorized.csv"
news_healines_vector_path = "./CleanData/NewsHeadlines_vectorized.csv"
# us_wildfires_2mil_path = "./CleanData/us_wildfires_2mil_cleaned.csv"

## Setting the filename
dm_state_total_area_filename = "dm_state_total_area_cleaned"
dm_state_percent_area_filename = "dm_state_percent_area_clean"
us_fires_burn_monthly_filename = "us_fires_burn_monthly"
or_weather_wildfires_filename = "or_weather_wildfires_cleaned"
lightining_fires_filename = "lightning_wildfires_clean"
or_weather_wildfires_comments_vector_filename = "or_weather_wildfires_cause_comments_vectorized"
or_weather_wildfires_specific_vector_filename = "or_weather_wildfires_specific_cause_vectorized"
news_healines_vector_filename = "NewsHeadlines_vectorized"
# us_wildfires_2mil_filename = "us_wildfires_2mil_cleaned"
#-------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------- 
def main():
    '''
    This function takes in the cleaned fire data, transforms it for unsupervised learning, performs kmeans clustering, and provides results
    '''
    print("\n############################################################################")
    print("\n---------- Ingesting Cleaned Fire and Drought Data ----------\n")
    # Ingest Data
    dm_state_total_area = pd.read_csv(dm_state_total_area_path)
    dm_state_percent_area = pd.read_csv(dm_state_percent_area_path)
    us_fires_burn_monthly = pd.read_csv(fires_monthly_folder_path)
    or_weather_wildfires = pd.read_csv(or_weather_wildfires_path)
    lightining_fires = pd.read_csv(lightining_fires_path)
    or_weather_wildfires_comments_vector = pd.read_csv(or_weather_wildfires_comments_vector_path)
    or_weather_wildfires_specific_vector = pd.read_csv(or_weather_wildfires_specific_vector_path)
    news_healines_vector = pd.read_csv(news_healines_vector_path)
    # us_wildfires_2mil = pd.read_csv(us_wildfires_2mil_path)
    
    print("\n---------- Transforming Fire Data for Unsupervised Learning ----------\n")
    ## Transform Data to keep numeric columns
    dm_state_total_area_transformed = transform_data(dm_state_total_area_filename, dm_state_total_area, ['None', 'D0', 'D1', 'D2', 'D3', 'D4', 'DSCI'])
    dm_state_percent_area_transformed = transform_data(dm_state_percent_area_filename, dm_state_percent_area, ['None', 'D0', 'D1', 'D2', 'D3', 'D4', 'DSCI'])
    us_fires_burn_monthly_transformed = transform_data(us_fires_burn_monthly_filename, us_fires_burn_monthly, ['Acres_Burned', 'Number_of_Fires', 'Acres_Burned_per_Fire'])
    or_weather_wildfires_transformed = transform_data(or_weather_wildfires_filename, or_weather_wildfires, ['tmax', 'tmin', 'tavg', 'prcp', 'EstTotalAcres', 'FireDuration_hrs'])
    lightining_fires_transformed = transform_data(lightining_fires_filename, lightining_fires, ["Days_to_extinguish_fire", "FIRE_SIZE"])
    # us_wildfires_2mil_transformed = transform_data(us_wildfires_2mil_filename, us_wildfires_2mil, ['FIRE_SIZE', 'FireDuration_hrs'])
    
    ## Transform Data to drop the label columnss
    or_weather_wildfires_comments_vector_transformed = or_weather_wildfires_comments_vector.drop(['GeneralCause'], axis=1)
    or_weather_wildfires_specific_vector_transformed = or_weather_wildfires_specific_vector.drop(['GeneralCause'], axis=1)
    news_healines_vector_transformed = news_healines_vector.drop(['LABEL'], axis=1)
    
    print("\n---------- Determining Optimal Number of Clusters ----------\n")
    #------------------------------------------------------------------------------------
    ## Elbow
    ## Ran these and then looked at the graphs. The values of k are then hard coded
    determine_k_elbow(dm_state_total_area_filename, dm_state_total_area_transformed)
    determine_k_elbow(dm_state_percent_area_filename, dm_state_percent_area_transformed)
    determine_k_elbow(us_fires_burn_monthly_filename, us_fires_burn_monthly_transformed)
    determine_k_elbow(or_weather_wildfires_filename, or_weather_wildfires_transformed)
    determine_k_elbow(lightining_fires_filename, lightining_fires_transformed)
    determine_k_elbow(or_weather_wildfires_comments_vector_filename, or_weather_wildfires_comments_vector_transformed)
    determine_k_elbow(or_weather_wildfires_specific_vector_filename, or_weather_wildfires_specific_vector_transformed)
    determine_k_elbow(news_healines_vector_filename, news_healines_vector_transformed)
    ## determine_k_elbow(us_wildfires_2mil_filename, us_wildfires_2mil_transformed)
    #------------------------------------------------------------------------------------
    
    #------------------------------------------------------------------------------------------------------------------------
    ## Silhouette
    determining_k_silhouette(dm_state_total_area_filename, dm_state_total_area_transformed)
    determining_k_silhouette(dm_state_percent_area_filename, dm_state_percent_area_transformed)
    determining_k_silhouette(us_fires_burn_monthly_filename, us_fires_burn_monthly_transformed)
    
    ## Create df sample with 70% of the rows from or_weather_wildfires_transformed
    # print(f"Sample from or_weather_wildfires_transformed")
    # or_weather_wildfires_transformed_sample = or_weather_wildfires_transformed.sample(frac=0.7)
    # or_weather_wildfires_transformed_sample.reset_index(drop=True, inplace=True)
    
    determining_k_silhouette(or_weather_wildfires_filename, or_weather_wildfires_transformed)
    determining_k_silhouette(lightining_fires_filename, lightining_fires_transformed)
    
    determining_k_silhouette(or_weather_wildfires_comments_vector_filename, or_weather_wildfires_comments_vector_transformed)
    determining_k_silhouette(or_weather_wildfires_specific_vector_filename, or_weather_wildfires_specific_vector_transformed)
    determining_k_silhouette(news_healines_vector_filename, news_healines_vector_transformed)
    
    # THIS TAKES WAYYYYYYYYY TO LONG - like I was on hour 18 and it was still not done...
    # determining_k_silhouette(us_wildfires_2mil_filename, us_wildfires_2mil_transformed)
    #------------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # print("\n---------- Performing Kmeans Clustering ----------\n")
    # ## Numeric Data
    by_hand_kmeans(dm_state_total_area_filename, dm_state_total_area_transformed, k=2)
    by_hand_kmeans(dm_state_percent_area_filename, dm_state_percent_area_transformed, k=2)
    by_hand_kmeans(us_fires_burn_monthly_filename, us_fires_burn_monthly_transformed, k=2)
    by_hand_kmeans(or_weather_wildfires_filename, or_weather_wildfires_transformed, k=2)
    by_hand_kmeans(lightining_fires_filename, lightining_fires_transformed, k=2)
    
    # ## Vector Text Data
    by_hand_kmeans(or_weather_wildfires_comments_vector_filename, or_weather_wildfires_comments_vector_transformed, k=2)
    by_hand_kmeans(or_weather_wildfires_specific_vector_filename, or_weather_wildfires_specific_vector_transformed, k=2)
    by_hand_kmeans(news_healines_vector_filename, news_healines_vector_transformed, k=2)
    
    # # by_hand_kmeans(us_wildfires_2mil_filename, us_wildfires_2mil_transformed, k=2)    # This one just doesn't make any sense
    #-------------------------------------------------------------------------------------------------------------------

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
    
    k_values = range(2, 11)
    silhouette_score_dict = {}

    for n_clusters in k_values:
        
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)

        ax1.set_xlim([-1, 1])
        ax1.set_ylim([0, len(df) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10, n_init=20)
        cluster_labels = clusterer.fit_predict(df)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(df, cluster_labels)
        silhouette_score_dict[n_clusters] = silhouette_avg
        print(f"\n--- For n_clusters = {n_clusters} \n--- The average silhouette_score is : {silhouette_avg}\n")

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(df, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title(f"The silhouette plot for the various clusters - {filename}")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.savefig(f"./CreatedVisuals/kmeans/Silhouette/{filename}_silhouette_{n_clusters}.png")
        plt.close()
    
    print("\n\n--------------------------------------------------")
    print(f"---------- {filename} ----------")
    print(f"\n--- Value of K:       Silhouette Score: ----")
    for k, score in silhouette_score_dict.items():
        print(f"--- {k}             {score} ---")
    print("--------------------------------------------------\n\n")
    

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