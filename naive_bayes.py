'''
Get ready to do some Naive Bayes

This analysis pulls some inspiration from Dr. Ami Gates Code for Naive Bayes
https://gatesboltonanalytics.com/?page_id=282
'''

import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
import seaborn as sns
import random as rd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from sklearn.naive_bayes import MultinomialNB


## Setting the folder path to the cleaned but not formatted data

fires_monthly_folder_path = "./CleanData/us_fires_burn_monthly.csv"
or_weather_wildfires_path = "./CleanData/or_weather_wildfires_cleaned.csv"
or_weather_wildfires_comments_vector_path = "./CleanData/or_weather_wildfires_cause_comments_vectorized.csv"
or_weather_wildfires_specific_vector_path = "./CleanData/or_weather_wildfires_specific_cause_vectorized.csv"
news_headlines_vector_path = "./CleanData/NewsHeadlines_vectorized.csv"

## Setting the filename

us_fires_burn_monthly_filename = "us_fires_burn_monthly"
or_weather_wildfires_filename = "or_weather_wildfires_cleaned"
or_weather_wildfires_comments_vector_filename = "or_weather_wildfires_cause_comments_vectorized"
or_weather_wildfires_specific_vector_filename = "or_weather_wildfires_specific_cause_vectorized"
news_headlines_vector_filename = "NewsHeadlines_vectorized"
#-------------------------------------------------------------------------

def main():
    '''
    This is the main function for the decision tree modeling and visualizations on wildfire data
    '''
    
    print("\n ---------- Ingesting Data ---------- \n")
    fires_monthly_df = pd.read_csv(fires_monthly_folder_path)
    or_weather_wildfires_df = pd.read_csv(or_weather_wildfires_path)
    or_weather_wildfires_comments_vector_df = pd.read_csv(or_weather_wildfires_comments_vector_path)
    or_weather_wildfires_specific_vector_df = pd.read_csv(or_weather_wildfires_specific_vector_path)
    news_headlines_df = pd.read_csv(news_headlines_vector_path)
    
    print("\n ---------- Reformatting Some of the Target Labels ---------- \n")
    fires_monthly_df = set_fire_months(fires_monthly_df, "Month") # returns df with added column "Season"
    
    news_headlines_df_selected = news_headlines_df[news_headlines_df['LABEL'].isin(["wildfire", "weather"])]
    news_headlines_df_selected2 = news_headlines_df[news_headlines_df['LABEL'].isin(["burn", "weather"])]
    news_headlines_df2 = set_new_generic_label(news_headlines_df, "LABEL")
    
    or_weather_wildfires_df["FireDuration_hrs"] = or_weather_wildfires_df["FireDuration_hrs"].apply(lambda x: 0 if x <= 0 else x)
    or_weather_wildfires_df[["tmax", "tmin", "tavg", "prcp"]] = or_weather_wildfires_df[["tmax", "tmin", "tavg", "prcp"]].apply(lambda x: (x-x.min())/ (x.max() - x.min()))  

    
    print("\n\n ---------- Selecting Train and Test Data ---------- \n")
    fires_monthly_train_labels, fires_monthly_train_df, fires_monthly_test_labels, fires_monthly_test_df = setup_train_test_data(fires_monthly_df, label_col="Month", cols_of_interst_plus_label=["Month","Acres_Burned", "Number_of_Fires", "Acres_Burned_per_Fire"])
    fires_season_train_labels, fires_season_train_df, fires_season_test_labels, fires_season_test_df = setup_train_test_data(fires_monthly_df, label_col="Season", cols_of_interst_plus_label=["Season","Acres_Burned", "Number_of_Fires", "Acres_Burned_per_Fire"])
    
    news_train_labels, news_train_df, news_test_labels, news_test_df = setup_train_test_data(news_headlines_df, "LABEL")
    news2_train_labels, news2_train_df, news2_test_labels, news2_test_df = setup_train_test_data(news_headlines_df2, "GenericLabel")
    news_select_train_labels, news_select_train_df, news_select_test_labels, news_select_test_df = setup_train_test_data(news_headlines_df_selected, "LABEL")
    news_select2_train_labels, news_select2_train_df, news_select2_test_labels, news_select2_test_df = setup_train_test_data(news_headlines_df_selected2, "LABEL")
    
    or_cause_train_labels, or_cause_train_df, or_cause_test_labels, or_cause_test_df = setup_train_test_data(or_weather_wildfires_comments_vector_df, "GeneralCause")
    or_specific_train_labels, or_specific_train_df, or_specific_test_labels, or_specific_test_df = setup_train_test_data(or_weather_wildfires_specific_vector_df, "GeneralCause")
    or_train_labels, or_train_df, or_test_labels, or_test_df = setup_train_test_data(or_weather_wildfires_df, label_col="Size_class", cols_of_interst_plus_label=["Size_class","tmax", "tmin", "tavg", "prcp", "FireDuration_hrs"])

    print("\n\n ---------- Naive Bayes ---------- \n")
    multinomial_naive_bayes(fires_monthly_train_labels, fires_monthly_train_df, fires_monthly_test_labels, fires_monthly_test_df, us_fires_burn_monthly_filename)
    multinomial_naive_bayes(fires_season_train_labels, fires_season_train_df, fires_season_test_labels, fires_season_test_df, f"{us_fires_burn_monthly_filename}_season")
    
    multinomial_naive_bayes(news_train_labels, news_train_df, news_test_labels, news_test_df, news_headlines_vector_filename)
    multinomial_naive_bayes(news2_train_labels, news2_train_df, news2_test_labels, news2_test_df, f"{news_headlines_vector_filename}_2")
    multinomial_naive_bayes(news_select_train_labels, news_select_train_df, news_select_test_labels, news_select_test_df, f"{news_headlines_vector_filename}_wildfires_weather")
    multinomial_naive_bayes(news_select2_train_labels, news_select2_train_df, news_select2_test_labels, news_select2_test_df, f"{news_headlines_vector_filename}_burn_weather")
    
    multinomial_naive_bayes(or_cause_train_labels, or_cause_train_df, or_cause_test_labels, or_cause_test_df, or_weather_wildfires_comments_vector_filename)
    multinomial_naive_bayes(or_specific_train_labels, or_specific_train_df, or_specific_test_labels, or_specific_test_df, or_weather_wildfires_specific_vector_filename)
    multinomial_naive_bayes(or_train_labels, or_train_df, or_test_labels, or_test_df, or_weather_wildfires_filename)

    
def set_fire_months(df, month_col):
    
    df['Season'] = df[month_col].apply(lambda x: 'Fire_Season' if x in ['August', 'July', 'June', 'September'] else 'Normal')
    return df


def set_new_generic_label(df, label_col):
    
    df2 = df.copy()
    
    df2['GenericLabel'] = df2[label_col].apply(lambda x: 'Fire' if x in ['wildfire', 'burn', 'fire'] else 'Weather')
    
    df2 = df2.drop(label_col, axis=1)
    
    return df2
    

def setup_train_test_data(df, label_col, cols_of_interst_plus_label=None, test_size=0.2, seed_val=1):

    if cols_of_interst_plus_label is None:
        df2 = df.copy()
    else:
        df2 = df[cols_of_interst_plus_label]
    
    rd.seed(seed_val)
    train_df, test_df = train_test_split(df2, test_size=test_size)
    
    print(f"\n\n Training Dataset before labels removed \n {train_df.head()} \n\n")
    
    print(f"\n\n Testing Dataset before labels removed \n {test_df.head()} \n\n")
    
    # Test
    test_labels = test_df[label_col]
    test_df = test_df.drop([label_col], axis=1) # remove labels
    
    # Train
    train_labels = train_df[label_col]
    train_df = train_df.drop([label_col], axis = 1) # remove labels
    
    return train_labels, train_df, test_labels, test_df


def multinomial_naive_bayes(train_labels, train_df, test_labels, test_df, filename):

    #Create the modeler
    MyModelNB = MultinomialNB()

    MyModelNB.fit(train_df, train_labels)
    naive_pred = MyModelNB.predict(test_df)
    # print(np.round(MyModelNB.predict_proba(test_df),2))

    print("\nThe prediction from NB is:")
    print(naive_pred)
    print("\nThe actual labels are:")
    print(test_labels)

    score = accuracy_score(test_labels, naive_pred)
    print(f"\nThe accurary is for {filename} is : {score}\n")

    # Show the confusion matrix
    cm = confusion_matrix(test_labels, naive_pred)
    print(f"\nThe confusion matrix for {filename} is:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=MyModelNB.classes_)
    plt.figure(figsize=(18, 15))
    disp.plot(cmap='magma')
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Confusion Matrix\n - {filename} -") 
    plt.tight_layout()
    plt.savefig(f"./CreatedVisuals/NaiveBayes/{filename}_cm.png")
    plt.close()

# DO NOT REMOVE!!!
if __name__ == "__main__":
    main()