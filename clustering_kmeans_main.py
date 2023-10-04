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
from sklearn.preprocessing import StandardScaler
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
    
    print("\n---------- Transforming Data for Unsupervised Learning ----------\n")
    ## Transform Data to keep numeric columns
    dm_state_total_area_transformed = transform_data(dm_state_total_area_filename, dm_state_total_area, ['None', 'D0', 'D1', 'D2', 'D3', 'D4', 'DSCI'])
    dm_state_percent_area_transformed = transform_data(dm_state_percent_area_filename, dm_state_percent_area, ['None', 'D0', 'D1', 'D2', 'D3', 'D4', 'DSCI'])
    us_fires_burn_monthly_transformed = transform_data(us_fires_burn_monthly_filename, us_fires_burn_monthly, ['Acres_Burned', 'Number_of_Fires', 'Acres_Burned_per_Fire'])
    or_weather_wildfires_transformed = transform_data(or_weather_wildfires_filename, or_weather_wildfires, ['tmax', 'tmin', 'tavg', 'prcp', 'EstTotalAcres', 'FireDuration_hrs'])
    lightining_fires_transformed = transform_data(lightining_fires_filename, lightining_fires, ["Days_to_extinguish_fire", "FIRE_SIZE"])
    # us_wildfires_2mil_transformed = transform_data(us_wildfires_2mil_filename, us_wildfires_2mil, ['FIRE_SIZE', 'FireDuration_hrs'])
    
    # ## Transform Data to drop the label columnss
    or_weather_wildfires_comments_vector_transformed = or_weather_wildfires_comments_vector.drop(['GeneralCause'], axis=1)
    or_weather_wildfires_specific_vector_transformed = or_weather_wildfires_specific_vector.drop(['GeneralCause'], axis=1)
    news_healines_vector_transformed = news_healines_vector.drop(['LABEL'], axis=1)
    
    print("\n---------- Printing the Prepared Datasets for Website Use ----------\n")
    print_df(dm_state_total_area_filename, dm_state_total_area_transformed)
    print_df(dm_state_percent_area_filename, dm_state_percent_area_transformed)
    print_df(us_fires_burn_monthly_filename, us_fires_burn_monthly_transformed)
    print_df(or_weather_wildfires_filename, or_weather_wildfires_transformed)
    print_df(lightining_fires_filename, lightining_fires_transformed)
    print_df(or_weather_wildfires_comments_vector_filename, or_weather_wildfires_comments_vector_transformed)
    print_df(or_weather_wildfires_specific_vector_filename, or_weather_wildfires_specific_vector_transformed)
    print_df(news_healines_vector_filename, news_healines_vector_transformed)
    
    
    print("\n---------- Normalizing Data for Unsupervised Learning ----------\n")
    dm_state_total_area_norm = normalize_df(dm_state_total_area_filename, dm_state_total_area_transformed)
    dm_state_percent_area_norm = normalize_df(dm_state_percent_area_filename, dm_state_percent_area_transformed)
    us_fires_burn_monthly_norm = normalize_df(us_fires_burn_monthly_filename, us_fires_burn_monthly_transformed)
    or_weather_wildfires_norm = normalize_df(or_weather_wildfires_filename, or_weather_wildfires_transformed)
    lightining_fires_norm = normalize_df(lightining_fires_filename, lightining_fires_transformed)
    
    print("\n---------- Standardizing Data for Unsupervised Learning ----------\n")
    dm_state_total_area_stan = standardize_df(dm_state_total_area_filename, dm_state_total_area_transformed)
    dm_state_percent_area_stan = standardize_df(dm_state_percent_area_filename, dm_state_percent_area_transformed)
    us_fires_burn_monthly_stan = standardize_df(us_fires_burn_monthly_filename, us_fires_burn_monthly_transformed)
    or_weather_wildfires_stan = standardize_df(or_weather_wildfires_filename, or_weather_wildfires_transformed)
    lightining_fires_stan = standardize_df(lightining_fires_filename, lightining_fires_transformed)
    
    
    print("\n---------- Determining Optimal Number of Clusters ----------\n")
    
    print("\n--- creating samples for quick computation ---\n")
    or_weather_wildfires_transformed_sample = get_df_sample(or_weather_wildfires_filename, or_weather_wildfires_transformed, 0.3)
    or_weather_wildfires_norm_sample = get_df_sample(or_weather_wildfires_filename + "_normalized", or_weather_wildfires_norm, 0.3)
    or_weather_wildfires_stan_sample = get_df_sample(or_weather_wildfires_filename + "_standardized", or_weather_wildfires_stan, 0.3)
    
    dm_state_total_area_transformed_sample = get_df_sample(dm_state_total_area_filename, dm_state_total_area_transformed, 0.7)
    dm_state_total_area_norm_sample = get_df_sample(dm_state_total_area_filename + "_normalized", dm_state_total_area_norm, 0.7)
    dm_state_total_area_stan_sample = get_df_sample(dm_state_total_area_filename + "_standardized", dm_state_total_area_stan, 0.7)
    
    dm_state_percent_area_transformed_sample = get_df_sample(dm_state_percent_area_filename, dm_state_percent_area_transformed, 0.7)
    dm_state_percent_area_norm_sample = get_df_sample(dm_state_percent_area_filename  + "_normalized", dm_state_percent_area_norm, 0.7)
    dm_state_percent_area_stan_sample = get_df_sample(dm_state_percent_area_filename  + "_standardized", dm_state_percent_area_stan, 0.7)
    
    us_fires_burn_monthly_transformed_sample = get_df_sample(us_fires_burn_monthly_filename, us_fires_burn_monthly_transformed, 0.7)
    us_fires_burn_monthly_norm_sample = get_df_sample(us_fires_burn_monthly_filename + "_normalized", us_fires_burn_monthly_norm, 0.7)
    us_fires_burn_monthly_stan_sample = get_df_sample(us_fires_burn_monthly_filename + "_standardized", us_fires_burn_monthly_stan, 0.7)
    
    lightining_fires_transformed_sample = get_df_sample(lightining_fires_filename, lightining_fires_transformed, 0.3)
    lightining_fires_norm_sample = get_df_sample(lightining_fires_filename + "_normalized", lightining_fires_norm, 0.3)
    lightining_fires_stan_sample = get_df_sample(lightining_fires_filename + "_standardized", lightining_fires_stan, 0.3)
    
    #-------------------------------------------------------------------------------------------------------------------
    ## Elbow
    ## Ran these and then looked at the graphs. The values of k are then hard coded
    # determine_k_elbow(dm_state_total_area_filename, dm_state_total_area_transformed)
    # determine_k_elbow(dm_state_percent_area_filename, dm_state_percent_area_transformed)
    # determine_k_elbow(us_fires_burn_monthly_filename, us_fires_burn_monthly_transformed)
    # determine_k_elbow(or_weather_wildfires_filename, or_weather_wildfires_transformed)
    # determine_k_elbow(lightining_fires_filename, lightining_fires_transformed)
    # determine_k_elbow(or_weather_wildfires_comments_vector_filename, or_weather_wildfires_comments_vector_transformed)
    # determine_k_elbow(or_weather_wildfires_specific_vector_filename, or_weather_wildfires_specific_vector_transformed)
    # determine_k_elbow(news_healines_vector_filename, news_healines_vector_transformed)
    # ## determine_k_elbow(us_wildfires_2mil_filename, us_wildfires_2mil_transformed)
    
    ## Elbow Norm
    # determine_k_elbow(dm_state_total_area_filename + "_norm", dm_state_total_area_norm)
    # determine_k_elbow(dm_state_percent_area_filename + "_norm", dm_state_percent_area_norm)
    # determine_k_elbow(us_fires_burn_monthly_filename + "_norm", us_fires_burn_monthly_norm)
    # determine_k_elbow(or_weather_wildfires_filename + "_norm", or_weather_wildfires_norm)
    # determine_k_elbow(lightining_fires_filename + "_norm", lightining_fires_norm)
    ## determine_k_elbow(us_wildfires_2mil_filename, us_wildfires_2mil_transformed)
    #-------------------------------------------------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------------------------------------------------
    ## Silhouette
    ## This first method of plotting the silhouette score results prints those finger graphs
    # determining_k_silhouette(dm_state_total_area_filename, dm_state_total_area_transformed)
    # determining_k_silhouette(dm_state_percent_area_filename, dm_state_percent_area_transformed)
    # determining_k_silhouette(us_fires_burn_monthly_filename, us_fires_burn_monthly_transformed)
    # determining_k_silhouette(dm_state_total_area_filename + "_norm", dm_state_total_area_norm)
    # determining_k_silhouette(dm_state_percent_area_filename + "_norm", dm_state_percent_area_norm)
    # determining_k_silhouette(us_fires_burn_monthly_filename + "_norm", us_fires_burn_monthly_norm)    
    # determining_k_silhouette(or_weather_wildfires_filename, or_weather_wildfires_transformed_sample)
    # determining_k_silhouette(lightining_fires_filename, lightining_fires_transformed)
    # determining_k_silhouette(or_weather_wildfires_comments_vector_filename, or_weather_wildfires_comments_vector_transformed)
    # determining_k_silhouette(or_weather_wildfires_specific_vector_filename, or_weather_wildfires_specific_vector_transformed)
    # determining_k_silhouette(news_healines_vector_filename, news_healines_vector_transformed)
    
    # THIS TAKES WAYYYYYYYYY TO LONG - like I was on hour 18 and it was still not done...
    # determining_k_silhouette(us_wildfires_2mil_filename, us_wildfires_2mil_transformed)
    
    ## This method of plotting the silhouette score results in a line graph - more readable
    # silhouette_line_graph(dm_state_total_area_filename, dm_state_total_area_transformed_sample)
    # silhouette_line_graph(dm_state_total_area_filename + "_normalized", dm_state_total_area_norm_sample)
    # silhouette_line_graph(dm_state_total_area_filename + "_standardized", dm_state_total_area_stan_sample)
    
    # silhouette_line_graph(dm_state_percent_area_filename, dm_state_percent_area_transformed_sample)
    # silhouette_line_graph(dm_state_percent_area_filename + "_normalized", dm_state_percent_area_norm_sample)
    # silhouette_line_graph(dm_state_percent_area_filename + "_standardized", dm_state_percent_area_stan_sample)
    
    # silhouette_line_graph(us_fires_burn_monthly_filename, us_fires_burn_monthly_transformed_sample)
    # silhouette_line_graph(us_fires_burn_monthly_filename + "_normalized", us_fires_burn_monthly_norm_sample)
    # silhouette_line_graph(us_fires_burn_monthly_filename + "_standardized", us_fires_burn_monthly_stan_sample)
    
    # silhouette_line_graph(or_weather_wildfires_filename, or_weather_wildfires_transformed_sample)
    # silhouette_line_graph(or_weather_wildfires_filename + "_normalized", or_weather_wildfires_norm_sample)
    # silhouette_line_graph(or_weather_wildfires_filename + "_standardized", or_weather_wildfires_stan_sample)
    
    # silhouette_line_graph(lightining_fires_filename, lightining_fires_transformed_sample)
    # silhouette_line_graph(lightining_fires_filename + "_normalized", lightining_fires_norm_sample)
    # silhouette_line_graph(lightining_fires_filename + "_standardized", lightining_fires_stan_sample)
    
    # silhouette_line_graph(or_weather_wildfires_comments_vector_filename, or_weather_wildfires_comments_vector_transformed)
    # silhouette_line_graph(or_weather_wildfires_specific_vector_filename, or_weather_wildfires_specific_vector_transformed)
    # silhouette_line_graph(news_healines_vector_filename, news_healines_vector_transformed)
    # #---------------------------------------------------------------------------------------------------------------------

    #--------------------------------------------------------------------------------------------------------------------
    print("\n---------- Performing Kmeans Clustering on Numeric Values ----------\n")
    
    ## Playing with different values of k
    # kmeans_clustering(dm_state_total_area_filename + "_2", dm_state_total_area_transformed, 2)
    # kmeans_clustering(dm_state_total_area_filename + "_3", dm_state_total_area_transformed, 3)
    # kmeans_clustering(dm_state_total_area_filename + "_4", dm_state_total_area_transformed, 4)
    # kmeans_clustering(dm_state_total_area_filename + "_2_normalized", dm_state_total_area_norm, 2)
    # kmeans_clustering(dm_state_total_area_filename + "_3_normalized", dm_state_total_area_norm, 3)
    # kmeans_clustering(dm_state_total_area_filename + "_4_normalized", dm_state_total_area_norm, 4)
    # kmeans_clustering(dm_state_total_area_filename + "_2_standardized", dm_state_total_area_stan, 2)
    # kmeans_clustering(dm_state_total_area_filename + "_3_standardized", dm_state_total_area_stan, 3)
    # kmeans_clustering(dm_state_total_area_filename + "_4_standardized", dm_state_total_area_stan, 4)
    
    ## Record Data
    # kmeans_clustering(dm_state_total_area_filename, dm_state_total_area_transformed, 2)
    # kmeans_clustering(dm_state_percent_area_filename, dm_state_percent_area_transformed, 2)
    
    # kmeans_clustering(us_fires_burn_monthly_filename, us_fires_burn_monthly_transformed, 2)
    # kmeans_clustering(us_fires_burn_monthly_filename + "_2_normalized", us_fires_burn_monthly_norm, 2)
    # kmeans_clustering(us_fires_burn_monthly_filename + "_3_normalized", us_fires_burn_monthly_norm, 3)
    # kmeans_clustering(us_fires_burn_monthly_filename + "_4_normalized", us_fires_burn_monthly_norm, 4)
    # kmeans_clustering(us_fires_burn_monthly_filename + "_5_normalized", us_fires_burn_monthly_norm, 5)
    # kmeans_clustering(us_fires_burn_monthly_filename + "_2_standardized", us_fires_burn_monthly_stan, 2)
    # kmeans_clustering(us_fires_burn_monthly_filename + "_3_standardized", us_fires_burn_monthly_stan, 3)
    # kmeans_clustering(us_fires_burn_monthly_filename + "_4_standardized", us_fires_burn_monthly_stan, 4)
    # kmeans_clustering(us_fires_burn_monthly_filename + "_5_standardized", us_fires_burn_monthly_stan, 5)
    
    # kmeans_clustering(or_weather_wildfires_filename, or_weather_wildfires_transformed, 2)
    # kmeans_clustering(or_weather_wildfires_filename + "_2_normalized", or_weather_wildfires_norm, 2)
    # kmeans_clustering(or_weather_wildfires_filename + "_3_normalized", or_weather_wildfires_norm, 3)
    # kmeans_clustering(or_weather_wildfires_filename + "_4_normalized", or_weather_wildfires_norm, 4)
    # kmeans_clustering(or_weather_wildfires_filename + "_5_normalized", or_weather_wildfires_norm, 5)
    # kmeans_clustering(or_weather_wildfires_filename + "_6_normalized", or_weather_wildfires_norm, 6)
    # kmeans_clustering(or_weather_wildfires_filename + "_7_normalized", or_weather_wildfires_norm, 7)
    # kmeans_clustering(or_weather_wildfires_filename + "_2_standardized", or_weather_wildfires_stan, 2)
    # kmeans_clustering(or_weather_wildfires_filename + "_3_standardized", or_weather_wildfires_stan, 3)
    # kmeans_clustering(or_weather_wildfires_filename + "_4_standardized", or_weather_wildfires_stan, 4)
    # kmeans_clustering(or_weather_wildfires_filename + "_5_standardized", or_weather_wildfires_stan, 5)
    # kmeans_clustering(or_weather_wildfires_filename + "_6_standardized", or_weather_wildfires_stan, 6)
    # kmeans_clustering(or_weather_wildfires_filename + "_7_standardized", or_weather_wildfires_stan, 7)
    
    
    # kmeans_clustering(lightining_fires_filename, lightining_fires_transformed, 2)
    
    ## Vector Text Data
    # kmeans_clustering(or_weather_wildfires_comments_vector_filename, or_weather_wildfires_comments_vector_transformed, 2)
    kmeans_clustering(or_weather_wildfires_comments_vector_filename + "_10", or_weather_wildfires_comments_vector_transformed, 10)
    
    # kmeans_clustering(or_weather_wildfires_specific_vector_filename, or_weather_wildfires_specific_vector_transformed, 2)
    # kmeans_clustering(or_weather_wildfires_specific_vector_filename + "_9", or_weather_wildfires_specific_vector_transformed, 9)
    
    # kmeans_clustering(news_healines_vector_filename, news_healines_vector_transformed, 2)
    
    # Ignore as the output is bad and it takes WAY too long
    # kmeans_clustering(us_wildfires_2mil_filename, us_wildfires_2mil_transformed, 2)
    #--------------------------------------------------------------------------------------------------------------------

    print("\n############################################################################\n")
        
    
#-----------------------------------------------------------------------------------------------------------------   


def kmeans_clustering(filename, df, n_clusters):
    print(f"\n--- Performing KMeans on {filename} ---")
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    df['cluster'] = kmeans.fit_predict(df)
    centroids = kmeans.cluster_centers_

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df.iloc[:, :-1])  # Exclude the 'cluster' column

    # Create a scatter plot to visualize the clusters in reduced dimension space
    plt.figure(figsize=(8, 6))

    # Scatter plot each cluster
    for cluster in range(n_clusters):
        cluster_data = pca_result[df['cluster'] == cluster]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster + 1}')

    # Add labels and legend
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'KMeans Clustering with PCA\n{filename}')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(f"./CreatedVisuals/kmeans/{filename}_kmeans.png")
    plt.close()
    

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

def  print_df(filename, df):
    print("\n ------------------------------------------------------------ ")
    print(f"{filename}:\n{df.head()}")
    print(" ------------------------------------------------------------ \n")
    

def normalize_df(filename, df):
    '''
    This function normalizes the data in a dataframe
    Args:
        - filename
        - dataframe
    Returns:
        - dataframe (normalized)
    '''
    print(f"--- Performing Normalization for {filename} ---")
    
    df_normalized = df.copy()
    
    for column in df_normalized.columns:
        min_val = df_normalized[column].min()
        max_val = df_normalized[column].max()
        if min_val == max_val:
            df_normalized[column] = 0  # Handle the case where all values are the same
        else:
            df_normalized[column] = (df_normalized[column] - min_val) / (max_val - min_val)
    
    print("\n ------------------------------ ")
    print(f"Normalized {filename}:\n{df_normalized.head()}")
    print(" ------------------------------ \n")
    
    return df_normalized


def standardize_df(filename, df):
    '''
    This function uses the sklearn library to implement a standardization to the dataframe
    Args:
        - filename
        - dataframe
    Returns:
        - dataframe with standardized columns
    '''
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    print("\n ------------------------------ ")
    print(f"Standardized {filename}:\n{df_standardized.head()}")
    print(" ------------------------------ \n")
    
    return df_standardized


def get_df_sample(filename, df, percent=0.5):
    '''
    This function takes a random sample of the a certain percent from a dataframe
    Args:
        - filename
        - dataframe
        - percent to take (default 50%)
    Returns:
        - sampled dataframe
    '''
    
    # ## Create df sample with 50% of the rows from or_weather_wildfires_transformed
    print(f"{percent} sample of {filename}")
    df_sample = df.sample(frac=0.5)
    df_sample.reset_index(drop=True, inplace=True)
    
    return df_sample
    
    

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
    plt.title(f'Elbow Method for Optimal k\n{filename}')
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

def silhouette_line_graph(filename, df):
    '''
    This graph gives another option for evaluating the silhouette scores of the datasets of different values of k.
    Args:
        - filename
        - dataframe (clean and preped)
    Return:
        - none but saves a graph to the folder ./CreatedVisual/kmeans/Silhouette
    '''
    print(f"--- Silhouette Line Graph for {filename} ---")
    k_values = range(2, 11)
    silhouette_avg = []
    for num_clusters in k_values:
        print(f"k = {num_clusters}")
        kmeans = KMeans(n_clusters=num_clusters, n_init=20)
        kmeans.fit(df)
        cluster_labels = kmeans.labels_
        silhouette_avg.append(silhouette_score(df, cluster_labels))
    
    plt.plot(k_values,silhouette_avg,'bx-')    
    plt.xlabel('Values of K') 
    plt.ylabel('Silhouette score') 
    plt.title(f'Silhouette analysis For Optimal k\n{filename}')   
    plt.savefig(f"./CreatedVisuals/kmeans/Silhouette/{filename}_silhouette_line.png")
    plt.close()
    
# DO NOT REMOVE!!!
if __name__ == "__main__":
    main()