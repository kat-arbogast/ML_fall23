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
from sklearn.svm import LinearSVC

from collections import Counter


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
    fires_monthly_sample_dict = setup_train_test_data(fires_monthly_df, label_col="Month", cols_of_interst_plus_label=["Month","Acres_Burned", "Number_of_Fires", "Acres_Burned_per_Fire"])
    fires_season_sample_dict = setup_train_test_data(fires_monthly_df, label_col="Season", cols_of_interst_plus_label=["Season","Acres_Burned", "Number_of_Fires", "Acres_Burned_per_Fire"])
    
    news_sample_dict = setup_train_test_data(news_headlines_df, "LABEL")
    news2_sample_dict = setup_train_test_data(news_headlines_df2, "GenericLabel")
    news_sample_dict_selected = setup_train_test_data(news_headlines_df_selected, "LABEL")
    news_sample_dict2 = setup_train_test_data(news_headlines_df_selected2, "LABEL")
    
    or_cause_sample_dict = setup_train_test_data(or_weather_wildfires_comments_vector_df, "GeneralCause")
    or_specific_sample_dict = setup_train_test_data(or_weather_wildfires_specific_vector_df, "GeneralCause")
    or_sample_dict = setup_train_test_data(or_weather_wildfires_df, label_col="Size_class", cols_of_interst_plus_label=["Size_class","tmax", "tmin", "tavg", "prcp"])
    or_sample_balanced_dict = train_test_OR_weather(or_weather_wildfires_df, label_col="Size_class", cols_of_interst_plus_label=["Size_class","tmax", "tmin", "tavg", "prcp"])
    
    
    print("\n\n ---------- Selecting Train and Test Data ---------- \n")
    run_svm(fires_monthly_sample_dict, us_fires_burn_monthly_filename)
    run_svm(fires_season_sample_dict, f"{us_fires_burn_monthly_filename}_season")
    
    run_svm(news_sample_dict, news_headlines_vector_filename)
    run_svm(news2_sample_dict, f"{news_headlines_vector_filename}_2")
    run_svm(news_sample_dict_selected, f"{news_headlines_vector_filename}_wildfires_weather")
    run_svm(news_sample_dict2, f"{news_headlines_vector_filename}_burn_weather")
    
    run_svm(or_cause_sample_dict, or_weather_wildfires_comments_vector_filename)
    run_svm(or_specific_sample_dict, or_weather_wildfires_specific_vector_filename)
    run_svm(or_sample_dict, or_weather_wildfires_filename)
    run_svm(or_sample_balanced_dict, f"{or_weather_wildfires_filename}_balanced")
    

     
    # print("\n\n ---------- Further Visuals ---------- \n")
    # months = ["January","February","March","April","May","June","July", "August","September","October","November","December"]
    # label_counts(fires_monthly_dict_pred, us_fires_burn_monthly_filename, months)
    # label_counts(fires_season_dict_pred, f"{us_fires_burn_monthly_filename}_season")
    
    # label_counts(news_dict_pred, news_headlines_vector_filename)
    # label_counts(news2_dict_pred,  f"{news_headlines_vector_filename}_2")
    # label_counts(news_dict_select_pred, f"{news_headlines_vector_filename}_wildfires_weather")
    # label_counts(news_dict_select2_pred, f"{news_headlines_vector_filename}_burn_weather")
    
    # label_counts(or_dict_cause_pred, or_weather_wildfires_comments_vector_filename)
    # label_counts(or_dict_specific_pred, or_weather_wildfires_specific_vector_filename)
    # label_counts(or_dict_pred, or_weather_wildfires_filename, ["A", "B", "C", "D", "E", "F", "G"])
    # label_counts(or_dict_balanced_pred, f"{or_weather_wildfires_filename}_balanced")

    
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


def train_test_OR_weather(df, label_col, cols_of_interst_plus_label):
    '''
    This function aims to give an example of what a balanced dataset would like like
    '''
    
    label_counts_total = Counter(list(df[label_col]))
    
    labels = list(label_counts_total.keys())
    counts = list(label_counts_total.values())
    
    min_count = min(counts)
    min_count1 = min_count //2
    min_count2 = min_count // 5
    
    if cols_of_interst_plus_label is None:
        df2 = df.copy()
    else:
        df2 = df[cols_of_interst_plus_label]
   
   
    data_label_A = df2[df2[label_col] == 'A']
    data_label_B = df2[df2[label_col] == 'B']
    data_label_C = df2[df2[label_col] == 'C']
    data_label_D = df2[df2[label_col] == 'D']
    data_label_E = df2[df2[label_col] == 'E']
    data_label_F = df2[df2[label_col] == 'F']
    data_label_G = df2[df2[label_col] == 'G']


    sample_A, _ = train_test_split(data_label_A, train_size=min_count1, stratify=data_label_A[label_col], random_state=42)
    sample_B, _ = train_test_split(data_label_B, train_size=min_count1, stratify=data_label_B[label_col], random_state=42)
    sample_C, _ = train_test_split(data_label_C, train_size=min_count1, stratify=data_label_C[label_col], random_state=42)
    sample_D, _ = train_test_split(data_label_D, train_size=min_count1, stratify=data_label_D[label_col], random_state=42)
    sample_E, _ = train_test_split(data_label_E, train_size=min_count1, stratify=data_label_E[label_col], random_state=42)
    sample_F, _ = train_test_split(data_label_F, train_size=min_count1, stratify=data_label_F[label_col], random_state=42)
    sample_G, _ = train_test_split(data_label_G, train_size=min_count1, stratify=data_label_G[label_col], random_state=42)
    
    
    A_test, _ = train_test_split(data_label_A, train_size=min_count2, stratify=data_label_A[label_col], random_state=123)
    B_test, _ = train_test_split(data_label_B, train_size=min_count2, stratify=data_label_B[label_col], random_state=123)
    C_test, _ = train_test_split(data_label_C, train_size=min_count2, stratify=data_label_C[label_col], random_state=123)
    D_test, _ = train_test_split(data_label_D, train_size=min_count2, stratify=data_label_D[label_col], random_state=123)
    E_test, _ = train_test_split(data_label_E, train_size=min_count2, stratify=data_label_E[label_col], random_state=123)
    F_test, _ = train_test_split(data_label_F, train_size=min_count2, stratify=data_label_F[label_col], random_state=123)
    G_test, _ = train_test_split(data_label_G, train_size=min_count2, stratify=data_label_G[label_col], random_state=123)
    
    # Combine the sampled data from each category into your training set
    training_set = pd.concat([sample_A, sample_B, sample_C, sample_D, sample_E, sample_F, sample_G])
    testing_set = pd.concat([A_test, B_test, C_test, D_test, E_test, F_test, G_test])

    # Test
    test_labels = testing_set[label_col]
    test_df = testing_set.drop([label_col], axis=1) # remove labels
    
    # Train
    train_labels = training_set[label_col]
    train_df = training_set.drop([label_col], axis = 1) # remove labels
    
    sample_dict = {
        "train_labels" : train_labels,
        "train_df" : train_df,
        "test_labels" : test_labels,
        "test_df" : test_df
    }

    return sample_dict


def label_counts(data_dict, filename, order=None):
        
    label_counts_pred = Counter(list(data_dict["naive_pred"]))
    label_counts_test = Counter(list(data_dict["test_labels"]))
    label_counts_train = Counter(list(data_dict["train_labels"]))
    
    labels_pred = list(label_counts_pred.keys())
    values_pred = list(label_counts_pred.values())
    
    labels_test = list(label_counts_test.keys())
    values_test = list(label_counts_test.values())
    
    labels_train = list(label_counts_train.keys())
    values_train = list(label_counts_train.values())
    
    total_pred = sum(values_pred)
    total_test = sum(values_test)
    total_train = sum(values_train)

    percentages_pred = [(value / total_pred) for value in values_pred]
    percentages_test = [(value / total_test) for value in values_test]
    percentages_train = [(value / total_train) for value in values_train]
    
    labels = list(set(labels_pred + labels_test + labels_train))
    
    pred_df = pd.DataFrame({
        "Label" : labels_pred,
        "Predicted" : percentages_pred
    })
    
    test_df = pd.DataFrame({
        "Label" : labels_test,
        "Test" : percentages_test
    })
    
    train_df = pd.DataFrame({
        "Label" : labels_train,
        "Train" : percentages_train
    })

    merged_df = train_df.merge(test_df, on='Label', how='outer').merge(pred_df, on='Label', how='outer')
    merged_df = merged_df.fillna(0)
    
    if order is not None:
        print(order)
        merged_df['Label'] = pd.Categorical(merged_df['Label'], categories=order, ordered=True)
        merged_df = merged_df.sort_values(by='Label')
        print(merged_df)
    
    plt.figure(figsize=(28, 10))
    ax = merged_df.plot(x="Label", y=["Test", "Predicted"], kind="bar", rot=0, color=["#364659", "#6C90D9"])
    plt.xticks(rotation=90, ha='right')
    plt.xlabel('\nLabel')
    plt.ylabel('Portion of Data with that Label\n')
    plt.title(f'Comparison of Naive Bayes Predictions and Actual Label Proportions\n{filename}')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
    
    plt.tight_layout()   
    plt.savefig(f"./CreatedVisuals/NaiveBayes/label_counts/{filename}_test_v_pred.png")
    plt.close()
    
    
    
    threshold = 1/len(list(set(labels)))
    plt.figure(figsize=(20, 10))
    ax = merged_df.plot(x="Label", y=["Train"], kind="bar", rot=0, color=["#8EA3BF"])
    plt.axhline(y=threshold, color='#D9CDBF', linestyle='--', label='Even Distribution')
    plt.xticks(rotation=90, ha='right')
    plt.xlabel('\nLabel')
    plt.ylabel('Portion of Data with that Label\n')
    plt.title(f'Proportion of Labels in the Training Set\n{filename}')
    
    plt.tight_layout()    
    plt.savefig(f"./CreatedVisuals/NaiveBayes/label_counts/{filename}_train.png")
    plt.close()


def run_svm(sample_dict, filename, visual_folder="./CreatedVisuals/svm"):
    
    
    
    ## Data
    train_labels = sample_dict["train_labels"]
    train_df = sample_dict["train_df"]
    test_labels = sample_dict['test_labels']
    test_df = sample_dict['test_df']
    

    SVM_Model1=LinearSVC(C=50, dual="auto")
    SVM_Model1.fit(train_df, train_labels)

    print("SVM 1 prediction:\n", SVM_Model1.predict(test_df))
    print("Actual:")
    print(test_labels)
    
    score = accuracy_score(test_labels, SVM_Model1.predict(test_df))
    print(f"\nThe accurary is for Linear SVM {filename} is : {score}\n")

    SVM_matrix = confusion_matrix(test_labels, SVM_Model1.predict(test_df))
    print("\nThe confusion matrix for Linear SVM is:")
    print(SVM_matrix)
    print("\n\n")
    
    disp = ConfusionMatrixDisplay(confusion_matrix=SVM_matrix, display_labels=SVM_Model1.classes_)
    plt.figure(figsize=(18, 15))
    disp.plot(cmap='magma')
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Confusion Matrix Linear SVM\n- {filename} Linear SVM -") 
    plt.tight_layout()
    plt.savefig(f"{visual_folder}/{filename}_linear_svm_cm.png")
    plt.close()
    
    
    #--------------other kernels
    ## RBF
    print("--- Starting RBF ---")
    SVM_Model2=sklearn.svm.SVC(C=1.0, kernel='rbf', gamma="auto")
    SVM_Model2.fit(train_df, train_labels)

    print("SVM prediction:\n", SVM_Model2.predict(test_df))
    print("Actual:")
    print(test_labels)
    
    score = accuracy_score(test_labels, SVM_Model2.predict(test_df))
    print(f"\nThe accurary is for rbf SVM {filename} is : {score}\n")

    SVM_matrix = confusion_matrix(test_labels, SVM_Model2.predict(test_df))
    print("\nThe confusion matrix for rbf SVM is:")
    print(SVM_matrix)
    print("\n\n")
    
    
    disp = ConfusionMatrixDisplay(confusion_matrix=SVM_matrix, display_labels=SVM_Model1.classes_)
    plt.figure(figsize=(18, 15))
    disp.plot(cmap='magma')
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Confusion Matrix RBF SVM\n- {filename} -") 
    plt.tight_layout()
    plt.savefig(f"{visual_folder}/{filename}_rbf_svm_cm.png")
    plt.close()

    # ## POLY
    # print("--- Starting Poly ---")
    # SVM_Model3=sklearn.svm.SVC(C=1.0, kernel='poly', degree=3, gamma="auto")
    # SVM_Model3.fit(train_df, train_labels)

    # print("SVM prediction:\n", SVM_Model3.predict(test_df))
    # print("Actual:")
    # print(test_labels)
    
    # score = accuracy_score(test_labels, SVM_Model3.predict(test_df))
    # print(f"\nThe accurary is for poly SVM {filename} is : {score}\n")

    # SVM_matrix = confusion_matrix(test_labels, SVM_Model3.predict(test_df))
    # print("\nThe confusion matrix for poly p = 3 SVM is:")
    # print(SVM_matrix)
    # print("\n\n")
    
    # disp = ConfusionMatrixDisplay(confusion_matrix=SVM_matrix, display_labels=SVM_Model1.classes_)
    # plt.figure(figsize=(18, 15))
    # disp.plot(cmap='magma')
    # plt.xticks(rotation=45, ha='right')
    # plt.title(f"Confusion Matrix\n - {filename} Linear SVM -") 
    # plt.tight_layout()
    # plt.savefig(f"{visual_folder}/{filename}_poly_svm_cm.png")
    # plt.close()


# DO NOT REMOVE!!!
if __name__ == "__main__":
    main()