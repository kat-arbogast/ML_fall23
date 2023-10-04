import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from clustering_kmeans_main import transform_data, normalize_df, standardize_df, print_df, get_df_sample


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

def main():
    
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
    
    print("\n---------- Performing DBSCAN CLustering ----------")
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
    
    
    
    ##################################################################################################################
    
    
    ## Playing with different values of k
    dbscan_clustering(dm_state_total_area_filename + "_2", dm_state_total_area_transformed, 2)
    # dbscan_clustering(dm_state_total_area_filename + "_3", dm_state_total_area_transformed, 3)
    # dbscan_clustering(dm_state_total_area_filename + "_4", dm_state_total_area_transformed, 4)
    # dbscan_clustering(dm_state_total_area_filename + "_2_normalized", dm_state_total_area_norm, 2)
    # dbscan_clustering(dm_state_total_area_filename + "_3_normalized", dm_state_total_area_norm, 3)
    # dbscan_clustering(dm_state_total_area_filename + "_4_normalized", dm_state_total_area_norm, 4)
    dbscan_clustering(dm_state_total_area_filename + "_2_standardized", dm_state_total_area_stan, 2)
    # dbscan_clustering(dm_state_total_area_filename + "_3_standardized", dm_state_total_area_stan, 3)
    # dbscan_clustering(dm_state_total_area_filename + "_4_standardized", dm_state_total_area_stan, 4)
    
    # ## Record Data
    # dbscan_clustering(dm_state_total_area_filename, dm_state_total_area_transformed, 2)
    # dbscan_clustering(dm_state_percent_area_filename, dm_state_percent_area_transformed, 2)
    
    # dbscan_clustering(us_fires_burn_monthly_filename, us_fires_burn_monthly_transformed, 2)
    # dbscan_clustering(us_fires_burn_monthly_filename + "_2_normalized", us_fires_burn_monthly_norm, 2)
    # dbscan_clustering(us_fires_burn_monthly_filename + "_2_standardized", us_fires_burn_monthly_stan, 2)
    
    # dbscan_clustering(or_weather_wildfires_filename, or_weather_wildfires_transformed, 2)
    # kmeans_cludbscan_clusteringstering(or_weather_wildfires_filename + "_2_normalized", or_weather_wildfires_norm, 2)
    # dbscan_clustering(or_weather_wildfires_filename + "_2_standardized", or_weather_wildfires_stan, 2)
    
    
    # dbscan_clustering(lightining_fires_filename, lightining_fires_transformed, 2)
    # dbscan_clustering(lightining_fires_filename + "_2_normalized", lightining_fires_norm, 2)
    # dbscan_clustering(lightining_fires_filename + "_2_standardized", lightining_fires_stan, 2)
    
    # ## Vector Text Data
    # dbscan_clustering(or_weather_wildfires_comments_vector_filename, or_weather_wildfires_comments_vector_transformed, 2)
    # dbscan_clustering(or_weather_wildfires_comments_vector_filename + "_8", or_weather_wildfires_comments_vector_transformed, 8)
    
    # dbscan_clustering(or_weather_wildfires_specific_vector_filename, or_weather_wildfires_specific_vector_transformed, 2)
    # dbscan_clustering(or_weather_wildfires_specific_vector_filename + "_9", or_weather_wildfires_specific_vector_transformed, 9)
    
    # dbscan_clustering(news_healines_vector_filename, news_healines_vector_transformed, 2)
    
    
    
    
    
    
    #################################################################################################################
    

def dbscan_clustering(filename, df, num_clusters, eps=0.5):
    print(f"--- Performing DBSCAN on {filename} ---")

    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=num_clusters)
    clusters = dbscan.fit_predict(df)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(df)
    
    # Create a DataFrame with PCA results and cluster labels
    pca_df = pd.DataFrame(data={'PCA1': pca_data[:, 0], 'PCA2': pca_data[:, 1], 'Cluster': clusters})
    
    # Create a scatter plot with cluster labels
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(pca_df['PCA1'], pca_df['PCA2'], c=pca_df['Cluster'], cmap='viridis')
    plt.title(f"DBCCAN CLustering\n{filename}")
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend(*scatter.legend_elements(), title='Clusters')

    # Save the plot
    plt.savefig(f"./CreatedVisuals/dbscan/{filename}_dbscan.png")
    plt.close()
    
# DO NOT REMOVE!!!
if __name__ == "__main__":
    main()