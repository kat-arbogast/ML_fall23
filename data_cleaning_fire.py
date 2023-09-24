import sqlite3
import pandas as pd

path_clean = './CleanData'
fires_monthly_folder_path = './DirtyData/Wildfire_Data/By_month_fires'
lightning_fires_path_dirty = './DirtyData/Wildfire_Data/lightining_fires_kaggle/US_Lightning_Forest_Fires.csv'
us_wildfires_2mil_path_clean = './CleanData/us_wildfires_2mil_cleaned.csv'


def main():
    '''
    This is the main function for cleaning the different datasets centered around the topic of fires and weather
    '''
    
    ## Fires Burned Monthly Datasets
    print("\n\n#################### Fires Burned Monthly Datasets ####################\n")
    fires_df_dict = read_in_fires_monthly(fires_monthly_folder_path)
    us_fires_burn_monthly = clean_month_fire_data(fires_df_dict)
    save_clean_to_csv(us_fires_burn_monthly, path_clean, "us_fires_burn_monthly.csv")
    
    
    ## Lightning Dataset
    print("\n\n#################### Lightning Fires Dataset ####################\n")
    lightning_df = lightning_fires_cleaning(lightning_fires_path_dirty)
    save_clean_to_csv(lightning_df, path_clean, "lightning_wildfires_clean.csv")
    
    
    ## US Wildfires 2 Million
    # from Kaggle
    '''
    Unfortunatly GitHub can not handle the size of the raw file, therefore this repository shows the code that 
    was applied to the dataframe, but does not actually call the functions that cleaned the raw data. 
    To access the raw data her is the link to the Kaggle page
    link: https://www.kaggle.com/datasets/braddarrow/23-million-wildfires
    '''
    print("\n\n#################### US Wildfires 2 Million Dataset ####################\n")
    
    # reading in the raw data - FOR LOCAL DEVICE USE - uses a connection to the database 
    # us_wildfires_2mil = database_connection('./Wildfire_Data/US_2mil_wildfires_kaggle/FPA_FOD_20221014.sqlite', 'Fires')  
    # us_wildfires_2mil_cleaned = cleaning_us_wildfires_2mil(us_wildfires_2mil)
    
    
    # reading in the cleaned data - FOR GITHUB USE - looks at what the cleaned datanow looks like
    us_wildfires_2mil_cleaned = pd.read_csv(us_wildfires_2mil_path_clean)
    
    year_list = list(us_wildfires_2mil_cleaned['FIRE_YEAR'].unique())      
    print(f'\nUnique Year in Full dataframe: \n{year_list}\n')
    
    print("\n--- US Wildfires 2 Million CLEANED INFO ---\n")
    print(f"\n\n{us_wildfires_2mil_cleaned.info()}\n")
    
    print("\n--- US Wildfires 2 Million CLEANED HEAD ---")
    print(f"\n\n{us_wildfires_2mil_cleaned.head()}\n")
    
    save_clean_to_csv(us_wildfires_2mil_cleaned, path_clean, 'us_wildfires_2mil_cleaned.csv')
    
    print("############################################################################")
    print("\n\nWOOOOOOOOOOO\n")

    
    


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
    print("\n--- Cleaning Monthly Fires Data ---\n")
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
    df2.sort_values(by=['Year'], inplace=True)
    year_list = list(df2['Year'].unique())              
    print(f'\nUnique Year in Full dataframe: \n{year_list}\n')
    
    print("\n--- Monthly US Fires CLEANED INFO ---\n")
    print(f"\n\n{df2.info()}\n")
    
    print("\n--- Monthly US Fires CLEANED HEAD ---")
    print(f"\n\n{df2.head()}\n")
    
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
    print("\n--- Cleaning Lightning Data ---\n")
    
    lightning_fires = pd.read_csv(input_dir)

    lightning_fires = lightning_fires.drop(columns=["Unnamed: 0", "index", "FIPS_CODE", "FIPS_NAME"])
    lightning_fires["Fire_Date"] = pd.to_datetime(lightning_fires["Fire_Date"])
    lightning_fires = lightning_fires.dropna()
    lightning_fires.sort_values(by=['Fire_Date'], inplace=True)

    year_list = list(lightning_fires['FIRE_YEAR'].unique())              
    print(f'\nUnique Year in Full dataframe: \n{year_list}\n')
    
    print("\n--- Lightning US Fires CLEANED INFO ---\n")
    print(f"\n\n{lightning_fires.info()}\n")
    
    print("\n--- Lightning US Fires CLEANED HEAD ---")
    print(f"\n\n{lightning_fires.head()}\n")
    
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
    print("\n--- Connecting to Database to collect US Wildfires 2 Million Dataset ---\n")
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
    print(f"\n********** Saving {filename} **********\n")
    df.to_csv(f"{output_dir}/{filename}", index=False)


def cleaning_us_wildfires_2mil(df):
    '''
    This is the function that is used on the local device to clean the raw US Wildfires 2 Million Dataset from Kaggle
    '''
    print()
    df_clean = df.copy()
    
    cols_to_drop = ['OBJECTID',
                    'Shape',
                    'FOD_ID',
                    'LOCAL_FIRE_REPORT_ID',
                    'LOCAL_INCIDENT_ID',
                    'ICS_209_PLUS_INCIDENT_JOIN_ID',
                    'ICS_209_PLUS_COMPLEX_JOIN_ID',
                    'MTBS_ID',
                    'NWCG_REPORTING_UNIT_ID',
                    'SOURCE_SYSTEM_TYPE',
                    'SOURCE_SYSTEM',
                    'NWCG_REPORTING_UNIT_NAME',
                    'SOURCE_REPORTING_UNIT',
                    'SOURCE_REPORTING_UNIT_NAME',
                    'FIRE_CODE',
                    'FIRE_NAME',
                    'MTBS_FIRE_NAME',
                    'COMPLEX_NAME',
                    'DISCOVERY_DOY',
                    'CONT_DOY',
                    'LATITUDE',
                    'LONGITUDE',
                    'FIPS_CODE',
                    'FIPS_NAME',
                    'FPA_ID'
                    ]
    
    df_clean = dropping_cols(df_clean, cols_to_drop)
    df_filtered = df_clean[df_clean['FIRE_YEAR'] >= 2000]
    df_filtered = cleaning_us_wildfires_2mil_dtypes(df_filtered)
    df_filtered = remove_na(df_filtered)
    df_filtered.sort_values(by=['DISCOVERY_DATE'], inplace=True)
    
    return df_filtered

def dropping_cols(df, cols_to_drop):
    '''
    This function removes the columns that are not needed for the analysis I wish to perform.
    This also will hopefully help cut down on the file size.
    Agrs:
        - dataframe
    Returns:
        - dataframe with only the remaining columns
    '''
    
    print("\n--- Removing Columns ---\n")
    df = df.drop(columns = cols_to_drop, axis=1)
    
    return df

def remove_na(df):
    print("\n--- Removing rows with NA values ---\n")
    
    df2 = df.copy()
    
    nan_count = df2.isna().sum()
    print("NA Counts before they are removed\n")
    print(f"----------------------------------------")
    print(f"SHAPE : {df2.shape}")
    print(f"NA count is: \n{nan_count}")
    print(f"----------------------------------------\n\n")
    
    df2 = df2.dropna()

    return df2


def cleaning_us_wildfires_2mil_dtypes(df):
    
    print("\n--- Converting Data Types ---\n")

    convert_dict = {'FIRE_YEAR':'category',
                    'FIRE_SIZE':'float',
                    'FIRE_SIZE_CLASS':'category',
                    'STATE':'category',
                    'NWCG_GENERAL_CAUSE':'category',
                    'NWCG_CAUSE_CLASSIFICATION':'category',
                    'NWCG_CAUSE_AGE_CATEGORY':'int8'
                }

    # Re-map values in NWCG_CAUSE_AGE_CATEGORY column so we can later remove na values safely
    df['NWCG_CAUSE_AGE_CATEGORY'] = df['NWCG_CAUSE_AGE_CATEGORY'].apply(lambda x: 1 if x == 'Minor' else 0)
    
    df['NWCG_GENERAL_CAUSE'] = df['NWCG_GENERAL_CAUSE'].str.strip()

    df = df.astype(convert_dict)
    df['DISCOVERY_DATE'] = pd.to_datetime(df['DISCOVERY_DATE'], format='%m/%d/%Y')
    df['CONT_DATE'] = pd.to_datetime(df['CONT_DATE'], format='%m/%d/%Y')
    
    return df
    

def sampling_us_wildfire_data(df):
    '''
    I first want to see if I clean it down firs that I won't have to sample for github
    '''
    pass

# DO NOT REMOVE!!!
if __name__ == "__main__":
    main()