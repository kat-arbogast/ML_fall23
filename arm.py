'''
Get ready for some Association Rule Mining!!!
'''
import pandas as pd
from apyori import apriori
from data_cleaning_fire import basket_word_column


#-----------------------------------------------------------------------------------------------------------
# Setting the folder path to the cleaned but not formatted data
or_weather_wildfires_comments_vector_path = "./CleanData/or_weather_wildfires_cause_comments_vectorized.csv"
or_weather_wildfires_specific_vector_path = "./CleanData/or_weather_wildfires_specific_cause_vectorized.csv"
news_healines_vector_path = "./CleanData/NewsHeadlines_vectorized.csv"

# Setting the filename
or_weather_wildfires_comments_vector_filename = "or_weather_wildfires_cause_comments"
or_weather_wildfires_specific_vector_filename = "or_weather_wildfires_specific_cause"
news_healines_vector_filename = "NewsHeadlines"
#-----------------------------------------------------------------------------------------------------------


def main():
    print("\n############################################################################")
    print("\n---------- Ingesting Cleaned Fire and Drought Data ----------\n")
    ## Ingest Data    
    or_weather_wildfires_comments_vector = pd.read_csv(or_weather_wildfires_comments_vector_path)
    or_weather_wildfires_specific_vector = pd.read_csv(or_weather_wildfires_specific_vector_path)
    news_healines_vector = pd.read_csv(news_healines_vector_path)
    
    print("\n---------- Transforming Data for Unsupervised Learning ----------\n")
    ## Transform Data from vector to basket to a single column of word lists
    or_weather_wildfires_comments_vector_transformed = words_to_list_column(or_weather_wildfires_comments_vector_filename, or_weather_wildfires_comments_vector)
    or_weather_wildfires_specific_vector_transformed = words_to_list_column(or_weather_wildfires_specific_vector_filename, or_weather_wildfires_specific_vector)
    news_healines_vector_transformed = words_to_list_column(news_healines_vector_filename, news_healines_vector)
    
    print("\n---------- Running ARM ----------\n")
    ## Run Apriori Algorithm on the Datasets
    run_apriori(or_weather_wildfires_comments_vector_filename, or_weather_wildfires_comments_vector_transformed)
    run_apriori(or_weather_wildfires_specific_vector_filename, or_weather_wildfires_specific_vector_transformed)
    run_apriori(news_healines_vector_filename, news_healines_vector_transformed)

    print("\n\n############################################################################\n")
    
    

def words_to_list_column(filename, df):
    '''
    This function takes a vectorized word dataframe
    runs the basket function to turn non zero values to the column name
    and then takes all the word in a single row nad appends them to a list within one column
    Args:
        - filename
        - text dataframe in vectorized form
    Returns:
        - dataframe for basket style format
    '''
    print(f"\n--- Convert Vector Text df to Basket for {filename} ---\n")
    
    basket_df = basket_word_column(df)

    label_column = basket_df.columns[0]
    word_columns = basket_df.columns[1:]

    basket_df['combined_words'] = basket_df.apply(lambda row: [row[col] for col in word_columns if row[col]], axis=1)
    new_df = basket_df[[label_column, 'combined_words']]
    
    print(f"Head of the two column dataframe: \n{new_df.head()}")
        
    
    return new_df


def run_apriori(filename, df, min_support = 0.05, min_confidence = 0.07, min_lift = 1, min_length = 2, max_length = 7):
    '''
    Using the library apyori, run the apriori association rule mining to get a list of rules 
    based on conditions: [min_support, min_confidence, min_lift, min_length, max_length]
    Args:
        - filename
        - dataframe (that is one column of labels and one column of a list of words)
        - min_support (default = )
        - min_confidence (default = )
        - min_lift (default = )
        - min_length (default = )
        - max_length (default = )
    Returns:
        - 
    '''
    
    print(f"\n--- Running Apriori on {filename} ---\n")

    trans = df.iloc[:, 1].tolist()
    print(len(trans))


    # Change the parameters till you get rules
    association_rules = apriori(trans, 
                                min_support = min_support, 
                                min_confidence = min_confidence, 
                                min_lift = min_lift, 
                                min_length = min_length, 
                                max_length = max_length
                                )
    association_results = list(association_rules)

    print(association_results)



# DO NOT REMOVE!!!
if __name__ == "__main__":
    main()