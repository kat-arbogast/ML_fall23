import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

path_clean = './CleanData'
fires_monthly_folder_path = './DirtyData/Wildfire_Data/By_month_fires'
lightning_fires_path_dirty = './DirtyData/Wildfire_Data/lightining_fires_kaggle/US_Lightning_Forest_Fires.csv'
us_wildfires_2mil_path_clean = './CleanData/us_wildfires_2mil_cleaned.csv'
or_fires_weather_path_dirty = './DirtyData/Wildfire_Data/oregon_wildfire_kaggle/Wildfire_Weather_Merged_new.csv'
dm_total_area_path_dirty = './DirtyData/Drought_Monitor_Data/dm_export_20000101_20221231_state_total_area_catergorical.csv'
dm_percent_area_path_dirty = './DirtyData/Drought_Monitor_Data/dm_export_20000101_20221231_state_percent_area_catergorical.csv'
dm_dsci_path_dirty = './DirtyData/Drought_Monitor_Data/dm_export_20000101_20221231_state_dsci.csv'

def main():
    '''
    This is the main function for cleaning the different datasets centered around the topic of fires and weather
    '''
    #------------------------------------------------------------------------------------------------------
    ## Fires Burned Monthly Datasets
    print("\n\n#################### Fires Burned Monthly Datasets ####################\n")
    fires_df_dict = read_in_fires_monthly(fires_monthly_folder_path)
    us_fires_burn_monthly = clean_month_fire_data(fires_df_dict)
    save_clean_to_csv(us_fires_burn_monthly, path_clean, "us_fires_burn_monthly.csv")
    #------------------------------------------------------------------------------------------------------
    
    
    #------------------------------------------------------------------------------------------------------
    ## Lightning Dataset
    print("\n\n#################### Lightning Fires Dataset ####################\n")
    lightning_df = lightning_fires_cleaning(lightning_fires_path_dirty)
    save_clean_to_csv(lightning_df, path_clean, "lightning_wildfires_clean.csv")
    #------------------------------------------------------------------------------------------------------
    
    
    #------------------------------------------------------------------------------------------------------
    ## US Wildfires 2 Million
    # from Kaggle
    '''
    Unfortunatly GitHub can not handle the size of the raw file, therefore this repository shows the code that 
    was applied to the dataframe, but does not actually call the functions that cleaned the raw data. 
    To access the raw data her is the link to the Kaggle page
    link: https://www.kaggle.com/datasets/braddarrow/23-million-wildfires
    '''
    print("\n\n#################### US Wildfires 2 Million Dataset ####################\n")
    ## reading in the raw data - FOR LOCAL DEVICE USE - uses a connection to the database 
    # us_wildfires_2mil = database_connection('./Wildfire_Data/US_2mil_wildfires_kaggle/FPA_FOD_20221014.sqlite', 'Fires')  
    # us_wildfires_2mil_cleaned = cleaning_us_wildfires_2mil(us_wildfires_2mil)
    
    ## reading in the cleaned data - FOR GITHUB USE - looks at what the cleaned datanow looks like
    us_wildfires_2mil_cleaned = pd.read_csv(us_wildfires_2mil_path_clean)
    
    print("\n--- US Wildfires 2 Million CLEANED INFO ---\n")
    print(f"\n\n{us_wildfires_2mil_cleaned.info()}\n")
    
    print("\n--- US Wildfires 2 Million CLEANED HEAD ---")
    print(f"\n\n{us_wildfires_2mil_cleaned.head()}\n")
        
    # save_clean_to_csv(us_wildfires_2mil_cleaned, "./Sampled_US_Wildfire_Data" , 'us_wildfires_2mil_cleaned.csv')
    #------------------------------------------------------------------------------------------------------
    
    
    #------------------------------------------------------------------------------------------------------
    ## Oregon Wildfires and Weather
    # from Kaggle
    print("\n\n#################### Oregon Wildfires and Weather Dataset ####################\n")
    or_weather_wildfires = pd.read_csv(or_fires_weather_path_dirty,dtype={'Cause_Comments' : 'str', 'DistFireNumber' : 'str'})
    or_weather_wildfires, cause_comments_vector, specific_cause_vector, cause_comments_basket, specific_cause_basket = cleaning_or_fires_weather(or_weather_wildfires)
    save_clean_to_csv(or_weather_wildfires, path_clean, 'or_weather_wildfires_cleaned.csv')
    save_clean_to_csv(cause_comments_vector, path_clean, 'or_weather_wildfires_cause_comments_vectorized.csv')
    save_clean_to_csv(specific_cause_vector, path_clean, 'or_weather_wildfires_specific_cause_vectorized.csv')
    save_clean_to_csv(cause_comments_basket, path_clean, 'or_weather_wildfires_cause_comments_basket.csv')
    save_clean_to_csv(specific_cause_basket, path_clean, 'or_weather_wildfires_specific_cause_basket.csv')
    #------------------------------------------------------------------------------------------------------
    
    
    #------------------------------------------------------------------------------------------------------
    ## Drought Montior Data
    # from U.S. Drought Monitor - National Drought Mitigation Center at the University of Nebraska-Lincoln, 
    # the United States Department of Agriculture, and the National Oceanic and Atmospheric Administration.
    print("\n\n#################### Drought Montior Data ####################\n")
    dm_state_total_area = pd.read_csv(dm_total_area_path_dirty)
    dm_state_percent_area = pd.read_csv(dm_percent_area_path_dirty)
    dm_state_dsci = pd.read_csv(dm_dsci_path_dirty)
    dm_state_total_area, dm_state_percent_area = cleaning_dm(dm_state_total_area, dm_state_percent_area, dm_state_dsci)
    save_clean_to_csv(dm_state_total_area, path_clean, 'dm_state_total_area_cleaned.csv')
    save_clean_to_csv(dm_state_percent_area, path_clean, 'dm_state_percent_area_clean.csv')
    #------------------------------------------------------------------------------------------------------
    
    
    print("\n\n#################### News API Data ####################\n")
    news_headlines_api = pd.read_csv("./CleanData/NewsHeadlines_vectorized.csv")
    news_headlines_api_basket = basket_word_column(news_headlines_api)
    save_clean_to_csv(news_headlines_api_basket, path_clean, 'NewsHeadlines_basket.csv')
    
    print("############################################################################")

    

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
    
    df_filtered["DISCOVERY_DATETIME"] = pd.to_datetime(df_filtered['DISCOVERY_DATE'].astype(str) + ' ' + df_filtered['DISCOVERY_TIME'].astype(str))
    df_filtered["CONT_DATETIME"] = pd.to_datetime(df_filtered['CONT_DATE'].astype(str) + ' ' + df_filtered['CONT_TIME'].astype(str))
    df_filtered = df_filtered.drop(['DISCOVERY_TIME', 'CONT_TIME', 'DISCOVERY_DATE', 'CONT_DATE', 'FIRE_YEAR'], axis=1)
    
    df_filtered = fire_duration_hrs(df_filtered, "DISCOVERY_DATETIME", "CONT_DATETIME")
    
    df_filtered.sort_values(by=['DISCOVERY_DATETIME'], inplace=True)
    
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
    df.loc[df['NWCG_CAUSE_AGE_CATEGORY'] == 'Minor', 'NWCG_CAUSE_AGE_CATEGORY'] = 1
    df.loc[df['NWCG_CAUSE_AGE_CATEGORY'] != 'Minor', 'NWCG_CAUSE_AGE_CATEGORY'] = 0            
    
    df = df.replace(r'\s+', ' ', regex=True)

    df = df.astype(convert_dict)
    df['DISCOVERY_DATE'] = pd.to_datetime(df['DISCOVERY_DATE'], format='%m/%d/%Y')
    df['CONT_DATE'] = pd.to_datetime(df['CONT_DATE'], format='%m/%d/%Y')
    
    df['DISCOVERY_TIME'] = pd.to_datetime(df['DISCOVERY_TIME'], format='%H%M', errors='coerce').dt.time
    df['CONT_TIME'] = pd.to_datetime(df['CONT_TIME'], format='%H%M', errors='coerce').dt.time
    
    return df


def cleaning_or_fires_weather(df):
    '''
    This function is the main function for cleaning the Oregon Fires and Weather dataset
        - drops columns that are not needed for analysis or have too many NA
        - remove rows with na values
        - fix data types
        - create a column that is the fires duration
    Args:
        - dataframe
    Returns
        - dataframe (cleaned)
        - TO DO: include the interesting text columns as returns for text data
    '''
    df_clean = df.copy()
    
    # Saving these to vectorize them
    cause_comments = df_clean.loc[df_clean[['GeneralCause', 'Cause_Comments']].notna().all(axis=1)]
    specific_cause = df_clean.loc[df_clean[['GeneralCause', 'SpecificCause']].notna().all(axis=1)]
    
    cause_comments_vector = vectorize_word_column(cause_comments, 'Cause_Comments')
    specific_cause_vector = vectorize_word_column(specific_cause, 'SpecificCause')
    
    cause_comments_basket = basket_word_column(cause_comments_vector)
    specific_cause_basket = basket_word_column(specific_cause_vector)
    
    cols_to_drop = ['Lat_DD',
                    'Long_DD',
                    'LatLongDD',
                    'FireName',
                    'FullFireNumber',
                    'ModifiedDate',
                    'FireEvent',
                    'Cause_Comments', # removing because it has a lot of NA values
                    'LandmarkLocation', # removing because it has a lot of NA values
                    'SpecificCause',
                    'UnitName'
                    ]
    
    df_clean = dropping_cols(df_clean, cols_to_drop)
    df_clean = remove_na(df_clean)
    df_clean = cleaning_or_dtypes(df_clean)
    df_clean = fire_duration_hrs(df_clean, "ReportDateTime", "Control_DateTime")
    
    df_clean.sort_values(by=['Date'], inplace=True)
    year_list = list(df_clean['Year'].unique())              
    # print(f'\nUnique Year in Full dataframe: \n{year_list}\n')
    
    # print("\n--- Oregon Wildfires and Weather CLEANED INFO ---\n")
    # print(f"\n\n{df_clean.info()}\n")
    
    # print("\n--- Oregon Wildfires and Weather CLEANED HEAD ---")
    # print(f"\n\n{df_clean.head()}\n")
    

    print("\n--- Oregon Wildfires and Weather Causes Vector HEAD ---")
    print(f"\n\n{cause_comments_vector.head()}\n")
    print("\n--- Oregon Wildfires and Weather Specific Vector HEAD ---")
    print(f"\n\n{specific_cause_vector.head()}\n")
    
    
    print("\n--- Oregon Wildfires and Weather Causes Basket HEAD ---")
    print(f"\n\n{cause_comments_basket.head()}\n")
    print("\n--- Oregon Wildfires and Weather Specific Basket HEAD ---")
    print(f"\n\n{specific_cause_basket.head()}\n")
    
    return df_clean, cause_comments_vector, specific_cause_vector, cause_comments_basket, specific_cause_basket


def cleaning_or_dtypes(df):
    cols_to_convert = ['ReportDateTime', 'Control_DateTime', 'Date', 'Ign_DateTime']
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_datetime)
    
    return df

def fire_duration_hrs(df, start_time_col, end_time_col):
    df['FireDuration_hrs'] = (df[end_time_col] - df[start_time_col]).dt.total_seconds() / 3600
    
    return df

def vectorize_word_column(df, column_name):
    '''
    This function uses the methods as shown by Dr. Gates on how to vectorize text data from a dataframe column
    Args:
        - dataframe with at least a label column and a wordy column
        - column that contains the sentance
    Returns:
        - a vectorized dataframe with a column for labels and 0/1 columns for the words
    '''
    print("--- Vectorizing Comment Columns ---")
    df2 = df.copy()

    wordLIST=[]
    GeneralCauseLIST=[]
    for nextWordList, nextGeneralCause in zip(df2[column_name], df2["GeneralCause"]):
        wordLIST.append(nextWordList)
        GeneralCauseLIST.append(nextGeneralCause)
         
    NewLIST=[]
    for element in wordLIST:
        AllWords=element.split(" ")
        NewWordsList=[]
        for word in AllWords:
            word=word.lower()
            if word in GeneralCauseLIST:
                pass
            else:
                NewWordsList.append(word)
        NewWords=" ".join(NewWordsList)
        NewLIST.append(NewWords)
        
    wordLIST=NewLIST

    ## Instantiate your CV
    MyCountV=CountVectorizer(
            input="content",  ## because we have a csv file
            lowercase=True, 
            stop_words = "english",
            max_features=75
            )

    ## Use your CV 
    MyDTM = MyCountV.fit_transform(wordLIST)  # create a sparse matrix
    ColumnNames=MyCountV.get_feature_names_out()

    ## Build the data frame
    MyDTM_DF=pd.DataFrame(MyDTM.toarray(),columns=ColumnNames)

    ## Convert the labels from list to df
    GeneralCause_DF = pd.DataFrame(GeneralCauseLIST,columns=['GeneralCause'])

    ##Save original DF - without the lables
    My_Orig_DF=MyDTM_DF

    ## Now - let's create a complete and labeled
    ## dataframe:
    dfs = [GeneralCause_DF, MyDTM_DF]

    final_df = pd.concat(dfs,axis=1, join='inner')
    
    return final_df

def basket_word_column(df):
    '''
    This is almost make a basket datset. Somewhere to start though
    Agrs:
        - vecotrized text dataframe
    Returns:
        - a dataframe where 1 is the columns name and 0 becomes an empty string
    '''
    print("\n--- Make a Column of Words into Transactional/Basket Data ---\n")
    df2 = df.copy()
    
    print(f"\n\nBEFORE BASKET HEAD: \n {df2.head()}")
    
    for col in df2.columns[1:]:
        df2[col] = df2[col].apply(lambda x: col if x != 0 else '')
        
    print(f"\n\nAFTER BASKET HEAD: \n {df2.head()}")
    
    df2 = df2.drop(df2.columns[0], axis=1)

    return df2

def cleaning_dm(df_total_area, df_percent_area, df_dsci):
    '''
    This function cleans the Drought Monitoring Data which gives the square miles of area in drought based on state
        - remove MapDate column as the same information is held in a different column
        - check and remove NA values
    Args:
        - dataframe
        - dataframe
        - dataframe
    Returns:
        - dataframe
        - dataframe
        - dataframe
    '''
    

    df_total_area = remove_na(df_total_area)
    df_total_area = cleaning_dm_dtypes(df_total_area)
    
    
    df_percent_area = remove_na(df_percent_area)
    df_percent_area = cleaning_dm_dtypes(df_percent_area)
    
    df_dsci = remove_na(df_dsci)
    df_dsci = state_names_to_abbr(df_dsci)
    
    df_total_area = df_total_area.merge(df_dsci, on=['StateAbbreviation', 'MapDate'], how='inner')
    df_percent_area = df_percent_area.merge(df_dsci, on=['StateAbbreviation', 'MapDate'], how='inner')
    
    df_total_area = dropping_cols(df_total_area, ["MapDate", "StatisticFormatID", "Name"])
    df_percent_area = dropping_cols(df_percent_area, ["MapDate", "StatisticFormatID", "Name"])
    
    for col in ["None", "D0", "D1", "D2", "D3", "D4"]:
        df_total_area[col] = df_total_area[col].str.replace(',', '').astype(float)
    
    print("\n--- DM Total Area INFO ---\n")
    print(f"\n\n{df_total_area.info()}\n")
    
    print("\n--- DM Total Area HEAD ---")
    print(f"\n\n{df_total_area.head()}\n")
    
    print("\n--- DM Percent Area INFO ---\n")
    print(f"\n\n{df_percent_area.info()}\n")
    
    print("\n--- DM Percent Area HEAD ---")
    print(f"\n\n{df_percent_area.head()}\n")
    
    
    return df_total_area, df_percent_area
    
    

def cleaning_dm_dtypes(df):
    try:
        df['ValidStart'] = pd.to_datetime(df['ValidStart'], format='%Y-%m-%d')
        df['ValidEnd'] = pd.to_datetime(df['ValidEnd'], format='%Y-%m-%d')
    except:
        df['MapDate'] = pd.to_datetime(df['MapDate'], format='%Y%m%d')
        
    return df

def state_names_to_abbr(df):
    
    state_abbr = {
        'Alabama': 'AL',
        'Alaska': 'AK',
        'Arizona': 'AZ',
        'Arkansas': 'AR',
        'California': 'CA',
        'Colorado': 'CO',
        'Connecticut': 'CT',
        'Delaware': 'DE',
        'Florida': 'FL',
        'Georgia': 'GA',
        'Hawaii': 'HI',
        'Idaho': 'ID',
        'Illinois': 'IL',
        'Indiana': 'IN',
        'Iowa': 'IA',
        'Kansas': 'KS',
        'Kentucky': 'KY',
        'Louisiana': 'LA',
        'Maine': 'ME',
        'Maryland': 'MD',
        'Massachusetts': 'MA',
        'Michigan': 'MI',
        'Minnesota': 'MN',
        'Mississippi': 'MS',
        'Missouri': 'MO',
        'Montana': 'MT',
        'Nebraska': 'NE',
        'Nevada': 'NV',
        'New Hampshire': 'NH',
        'New Jersey': 'NJ',
        'New Mexico': 'NM',
        'New York': 'NY',
        'North Carolina': 'NC',
        'North Dakota': 'ND',
        'Ohio': 'OH',
        'Oklahoma': 'OK',
        'Oregon': 'OR',
        'Pennsylvania': 'PA',
        'Rhode Island': 'RI',
        'South Carolina': 'SC',
        'South Dakota': 'SD',
        'Tennessee': 'TN',
        'Texas': 'TX',
        'Utah': 'UT',
        'Vermont': 'VT',
        'Virginia': 'VA',
        'Washington': 'WA',
        'West Virginia': 'WV',
        'Wisconsin': 'WI',
        'Wyoming': 'WY',
        'Puerto Rico': 'PR'
    }
    
    df['StateAbbreviation'] = df['Name'].replace(state_abbr)
    
    return df
    
# DO NOT REMOVE!!!
if __name__ == "__main__":
    main()