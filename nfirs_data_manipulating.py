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
            print(f'./{new_folder}/{filename}/{i}')
            i_split = i.rsplit('.', 1)[0]
            data_path_list.append(f'./{new_folder}/{filename}/{i_split}')
            with ZipFile(f'./{new_folder}/{filename}/{i}', 'r') as f:
                #extract in same directory
                f.extractall(f'./{new_folder}/{filename}/{i_split}')
    
    return data_path_list

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


old_folder_2019 = 'Gregory_Adams_Wildfire_Data'
new_folder_2019 = 'usa_nfirs_data'
filename_2019 = 'usfa_nfirs_2019'


data_path_list_2019 = extracting_data(filename_2019, old_folder_2019, new_folder_2019)

total_file_list_2019 = []

for i in data_path_list_2019:
    file_list = get_folder_file_list(i)
    total_file_list_2019.append(file_list)

# for i in total_file_list_2019:
#     for j in i:
#         print(j)

print(f'{data_path_list_2019[1]}/{total_file_list_2019[-1][0]}')

causes_2019 = pd.read_csv(f'{data_path_list_2019[1]}/{total_file_list_2019[-1][0]}', sep = '^')

print(causes_2019.head(n=5))