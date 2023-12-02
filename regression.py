'''
Get ready to do some support vecotr machine analysis
'''

import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
import seaborn as sns
import random as rd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score


## Setting the folder path to the cleaned but not formatted data
dm_state_total_area_path = "./CleanData/dm_state_total_area_cleaned.csv"
dm_state_percent_area_path = "./CleanData/dm_state_percent_area_clean.csv"
fires_monthly_folder_path = "./CleanData/us_fires_burn_monthly.csv"
or_weather_wildfires_path = "./CleanData/or_weather_wildfires_cleaned.csv"

## Setting the filename
dm_state_total_area_filename = "dm_state_total_area_cleaned"
dm_state_percent_area_filename = "dm_state_percent_area_clean"
us_fires_burn_monthly_filename = "us_fires_burn_monthly"
or_weather_wildfires_filename = "or_weather_wildfires_cleaned"
#-------------------------------------------------------------------------

def main():
    '''
    This is the main function for the decision tree modeling and visualizations on wildfire data
    '''
    
    print("\n ---------- Ingesting Data ---------- \n")
    dm_state_total_area_df = pd.read_csv(dm_state_total_area_path)
    dm_state_percent_area_df = pd.read_csv(dm_state_percent_area_path)
    fires_monthly_df = pd.read_csv(fires_monthly_folder_path)
    or_weather_wildfires_df = pd.read_csv(or_weather_wildfires_path)
    
    
    print("\n ---------- Reformatting Some of the Target Labels ---------- \n")
    fires_monthly_df = set_fire_months(fires_monthly_df, "Month") # returns df with added column "Season"
    
    print("\n\n ---------- Selecting Train and Test Data ---------- \n")
    fires_monthly_sample_dict = setup_train_test_data(fires_monthly_df, label_col="Month", cols_of_interst_plus_label=["Month","Acres_Burned", "Number_of_Fires", "Acres_Burned_per_Fire"])
    fires_season_sample_dict = setup_train_test_data(fires_monthly_df, label_col="Season", cols_of_interst_plus_label=["Season","Acres_Burned", "Number_of_Fires", "Acres_Burned_per_Fire"])
    # Need to change the Size_class to fire_duration
    # or_sample_dict = setup_train_test_data(or_weather_wildfires_df, label_col="Size_class", cols_of_interst_plus_label=["Size_class","tmax", "tmin", "tavg", "prcp"])
    

   
def set_fire_months(df, month_col):
    
    df['Season'] = df[month_col].apply(lambda x: 'Fire_Season' if x in ['August', 'July', 'June', 'September'] else 'Normal')
    return df
    

def setup_train_test_data(df, label_col, cols_of_interst_plus_label=None, test_size=0.2, seed_val=1):

    if cols_of_interst_plus_label is None:
        df2 = df.copy()
    else:
        df2 = df[cols_of_interst_plus_label]
    
    rd.seed(seed_val)
    train_df, test_df, train_labels, test_labels = train_test_split(df2.drop(label_col, axis=1), df2[label_col], test_size=test_size, stratify=df2[label_col])
    
    print(f"\n\n Training Dataset with labels removed \n {train_df.head()} \n\n")
    
    print(f"\n\n Testing Dataset with labels removed \n {test_df.head()} \n\n")
    
    sample_dict = {
        "train_labels" : train_labels,
        "train_df" : train_df,
        "test_labels" : test_labels,
        "test_df" : test_df
    }
    
    return sample_dict


# DO NOT REMOVE!!!
if __name__ == "__main__":
    main()