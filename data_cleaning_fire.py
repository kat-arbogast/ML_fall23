import sqlite3
import pandas as pd

path_clean = './CleanData'
fires_monthly_folder_path = './DirtyData/Wildfire_Data/By_month_fires'
lightning_fires_path_dirty = './DirtyData/Wildfire_Data/lightining_fires_kaggle/US_Lightning_Forest_Fires.csv'


def main():
    '''
    This is the main function for cleaning the different datasets centered around the topic of fires and weather
    '''
    
    ## Fires Burned Monthly Datasets
    print("\n\n ---------- Fires Burned Monthly Datasets ---------- \n")
    fires_df_dict = read_in_fires_monthly(fires_monthly_folder_path)
    us_fires_burn_monthly = clean_month_fire_data(fires_df_dict)
    save_clean_to_csv(us_fires_burn_monthly, path_clean, "us_fires_burn_monthly.csv")
    
    ## Lightning Dataset
    print("\n\n ---------- Lightning Fires Dataset ---------- \n")
    lightning_df = lightning_fires_cleaning(lightning_fires_path_dirty)
    save_clean_to_csv(lightning_df, path_clean, "lightning_wildfires_clean.csv")


def read_in_fires_monthly(fires_monthly_folder_path):
    '''
    This reads in each of the csv files for the acres burned and number of fires per month and puts them into a dictionary fo dataframes
    
    Args:
        - relative path to the folder containing the monthly csvs
    Returns:
        - dictionary with each months df
    '''
    ## Per Month Wildfire Data
    df_month_dict = {
        "January" : pd.read_csv(f'{fires_monthly_folder_path}/US_Wildfires_January.csv', skiprows=1),
        "February": pd.read_csv(f'{fires_monthly_folder_path}/US_Wildfires_February.csv', skiprows=1),
        "March" : pd.read_csv(f'{fires_monthly_folder_path}/US_Wildfires_March.csv', skiprows=1),
        "April" : pd.read_csv(f'{fires_monthly_folder_path}/US_Wildfires_April.csv', skiprows=1),
        "May" : pd.read_csv(f'{fires_monthly_folder_path}/US_Wildfires_May.csv', skiprows=1),
        "June" : pd.read_csv(f'{fires_monthly_folder_path}/US_Wildfires_June.csv', skiprows=1),
        "July" : pd.read_csv(f'{fires_monthly_folder_path}/US_Wildfires_July.csv', skiprows=1),
        "August" : pd.read_csv(f'{fires_monthly_folder_path}/US_Wildfires_August.csv', skiprows=1),
        "September" : pd.read_csv(f'{fires_monthly_folder_path}/US_Wildfires_September.csv', skiprows=1),
        "October" : pd.read_csv(f'{fires_monthly_folder_path}/US_Wildfires_October.csv', skiprows=1),
        "November" : pd.read_csv(f'{fires_monthly_folder_path}/US_Wildfires_November.csv', skiprows=1),
        "December" : pd.read_csv(f'{fires_monthly_folder_path}/US_Wildfires_December.csv', skiprows=1)
    }
    return df_month_dict

def clean_month_fire_data(df_dict):
    '''
    This function takes the dictionary of data frames and:
        - adds a month column
        - concatonates all of the dataframes together
        - removes the year 2023 (The year has yet to finish)
        - makes year a object instead of a numeric
    Args:
        - dictionary of dataframes
    Returns:
        - dataframe (cleaned)
    '''
    for key, value in df_dict.items():
        value["Month"] = key
    
    df = pd.concat(df_dict.values(), axis=0)

    df = df.rename(columns={"Date":"Year", 
                            "Number of Fires" : "Number_of_Fires", 
                            "Acres Burned per Fire" : "Acres_Burned_per_Fire", 
                            "Acres Burned" : "Acres_Burned"})
    df["Year"] = df["Year"].astype("object")
    df2 = df.copy()
    
    # Removing 2023 because this data is not complete for all months yet - Project being done in 2023
    df2 = df2[df2["Year"] != 2023]
    year_list = list(df2['Year'].unique())
                     
    print(f'\nUnique Year in Full dataframe: \n{year_list}\n')
    print(f"\n{df2.info()}")
    
    return df2

    
def lightning_fires_cleaning(input_dir):
    '''
    This function cleans the lightning fires dataset from kaggle:
        - drops unnecessary or highly NA valued columns
        - drops the rows with NAs
    Args:
        - path to the csv file
    Returns:
        - dataframe (cleaned)
    '''

    lightning_fires = pd.read_csv(input_dir)

    lightning_fires = lightning_fires.drop(columns=["Unnamed: 0", "index", "FIPS_CODE", "FIPS_NAME"])
    lightning_fires["Fire_Date"] = pd.to_datetime(lightning_fires["Fire_Date"])
    lightning_fires = lightning_fires.dropna()

    print(lightning_fires.info())
    
    return lightning_fires


def database_connection(relative_path, table_name):
    '''
    Reads in data from a sql query or sqlite file
    Args:
        - realtive path to the file
        - the table's name that one wishes to query
    Returns:
        - dataframe of the table
    '''
    
    cnx = sqlite3.connect(relative_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", cnx)
    cnx.close()
    
    return df


def save_clean_to_csv(df, output_dir, filename):
    '''
    Saves the cleaned dataframes as a csv into a specified folder
    Args:
        - dataframe (cleaned)
        - output_dir (the relative path to the folder one wishes to save to)
        - filename (name of the .csv file with the extension!)
    '''
    df.to_csv(f"{output_dir}/{filename}", index=False)

    

# DO NOT REMOVE!!!
if __name__ == "__main__":
    main()