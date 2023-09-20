# This is where I will attempt to get the information from the wildfire information fromt the Gregory Adams email information
# Wish me the best of luck lol

import pandas as pd
import numpy as np 
from zipfile import ZipFile
import os


# FIRST THINGS FIRST - trying to read in  one and only one (baby steps) txt data value


def extracting_data(filename, old_folder, new_folder):
    '''
    This function will unzip data file. Look in that data file, unzip further files
    
    Args:
        - the zipfiles name
        - the current location of the zipfile
        - the new folder location for the extracted data to live in
    
    Returns:
        - a list of the paths to the inner folders that contain the text data
    '''
    
    print(f" -------- Beginning to Extract Data for {filename} -------- ")
    
    try:
        with ZipFile(f'./{old_folder}/{filename}.zip', 'r') as f:
            #extract in different directory
            f.extractall(f'./{new_folder}')
    except:
        print(f"The zip file {file_name} has most likely already been extracted.")
        

    # directory/folder path
    dir_path = f'./{new_folder}/{filename}'

    res = get_folder_file_list(dir_path)    
            
    data_path_list = []
    
    
    for i in res:
        if i.endswith('.zip'):
            print(f" -------- Extracting Data from inner zip folder {i} -------- ")
            i_split = i.rsplit('.', 1)[0]
            data_path_list.append(f'./{new_folder}/{filename}/{i_split}')
            try: 
                with ZipFile(f'./{new_folder}/{filename}/{i}', 'r') as f:
                    #extract in same directory
                    f.extractall(f'./{new_folder}/{filename}/{i_split}')
            except:
                print(f"Data from {i} has mostly likely already been extracted.")
    
    return data_path_list


def get_total_file_list(data_path_list):
    '''
    Loops through the data_path_list that is the directory paths of the inner data folders
    '''
    total_file_list = []

    for i in data_path_list:
        file_list = get_folder_file_list(i)
        total_file_list.append(file_list)
    
    return total_file_list
    

def get_folder_file_list(dir_path):
    '''
    Goes through a directory and appends the file names to a list
    
    Agrs:
        - relative directory path
    
    Returns:
        - list of file names withint the agrument directory
    '''
    # list to store files
    res = []

    # Iterate directory
    for file_path in os.listdir(dir_path):
        # check if current file_path is a file
        if os.path.isfile(os.path.join(dir_path, file_path)):
            # add filename to list
            res.append(file_path)
    
    return res

def clean_causes(df, clean_folder, clean_file_name):
    print("\n -------- STARTING TO CLEAN CAUSES DF --------\n")

    # Getting the date from the INCIDENT_KEY
    print("\n -------- GETTING THE INCIDENT DATE INTO A DATE TIME OBJECT -------- \n")
    df[['US_State', 'Fire_Department_ID', 'Date', 'incident_num', 'exp_no_2']] = df['INCIDENT_KEY'].str.split('_', expand=True)
    df = df.drop(columns=['Fire_Department_ID', 'US_State', 'incident_num', 'exp_no_2'])
    df["Date"] = df["Date"].astype(str)

    df["Date"] = df["Date"].str[0:2] + "-" + df["Date"].str[2:4] + "-" + df["Date"].str[4:]
    df["Date"] = pd.to_datetime(df["Date"], format='%m-%d-%Y')
    df = df.drop(columns = ["INC_DATE"])


    print("\n -------- CHANGING THE CAUSE CODES TO BE OBJECTS AND NOT INTs --------\n")
    df["EXP_NO"] = df["EXP_NO"].astype("object")
    df["PCC"] = df["PCC"].astype("object")
    df["CAUSE_CODE"] = df["CAUSE_CODE"].astype("object")
    df["GCC"] = df["GCC"].astype("object")

    print("\n\n\n")
    print(df.info())
    print(df.head(n=5))
    print("\n\n\n")


    df.to_csv(f"./CleanData/{clean_folder}/{clean_file_name}.csv", index=False)
    
    return df


old_folder_RAW = 'DirtyData/NFIRS_Wildfire_Data'
new_folder = 'CleanData/NFIRS_Wildfire_Data'
filename_2019 = 'usfa_nfirs_2019'
filename_2020 = 'nfirs_all_incident_pdr_2020'

## Extracting Data is for my local device where the raw data can be handled without size limits
# data_path_list_2019 = extracting_data(filename_2019, old_folder_RAW, new_folder)
# data_path_list_2020 = extracting_data(filename_2020, old_folder_RAW, new_folder)

# total_file_list_2019 = get_total_file_list(data_path_list_2019)
# total_file_list_2020 = get_total_file_list(data_path_list_2020)


# print("\n\n\n")
# print(" ----------------------- Data Path List ------------------------ ")

# for p in data_path_list_2020:
#     print(p)
    
# print("\n\n\n")
# print(" ----------------------- Total File Path List ------------------------ ")

# print(f"Length of the total_file_list_2019 {len(total_file_list_2020)}")

# for t in total_file_list_2020:
#     print(t)
        
# print("\n\n\n")
# print(" ----------------------- Causes Path ------------------------ \n")
# print(f'{data_path_list_2020[1]}/{total_file_list_2020[1][0]}')

# causes_2020 = pd.read_csv(f'{data_path_list_2020[1]}/{total_file_list_2020[1][0]}', sep = '^')



##  Hard coded for quicker cleaning and for the sake of reproducibility in the GitHub
causes_2020 = pd.read_csv('./DirtyData/NFIRS_Wildfire_Data/nfirs_all_incidents_2020/causes.txtt', sep="^")

causes_2020_clean = clean_causes(causes_2020.copy(), clean_folder= "NFIRS_Wildfire_Data/nfirs_2020", clean_file_name= "causes_2020_clean")