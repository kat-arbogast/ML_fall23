'''
The Problem 

Suppose you work for Amazon and they ask you to 
    (1) determine if a User should get a credit card (yes or no), and 
    (2) determine which products to advertise to the User. 

Imagine that you have all of Amazon User data, anything you need. 
Design/create a small dataset that you can use to address the questions above. 
    Paste it in the Word Document. 
    Keep it small with 3 to 4 columns and 25 to 30 rows. 
    You decide the column/variable names and what the data would look like.
    It can be anything you want that also makes sense with respect to the question. You create the dataset you need.

'''

import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
import seaborn as sns
import random as rd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import graphviz 

import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


amazon_user_df = pd.DataFrame({
    "Credit_Score": [600, 810, 820, 830, 840, 850, 610, 805, 815, 825, 835, 700, 600, 610, 620, 630, 640, 650, 610, 605, 615, 625, 635, 645, 655],
    "Income_Level": [90000, 95000, 100000, 15000, 110000, 115000, 95000, 92000, 98000, 102000, 108000, 91000, 30000, 35000, 40000, 45000, 59000, 55000, 35000, 80000, 38000, 42000, 48000, 90000, 56000],
    "Length_of_Credit_History": [8, 9, 10, 1, 12, 13, 9, 8, 10, 11, 2, 13, 2, 3, 4, 5, 6, 7, 3, 2, 14, 5, 6, 17, 8],
    'Products': [
        ['Wireless Router', 'Fitness Resistance Bands', 'Camera', 'Yoga Mat', 'Water Bottle', 'Fitness Tracker'],
        ['Gaming Mouse', 'Espresso Machine', 'Bluetooth Keyboard', 'Coffee', 'Coffee Maker', 'Noise-Canceling Headphones'],
        ['Camera', 'Travel Luggage', 'Electric Scooter'],
        ['Projector', 'Smart Light Bulbs', 'Weighted Blanket', 'White Noise Machine', 'Noise-Canceling Headphones'],
        ['Hiking Boots', 'Water Bottle', 'Backpack'],
        ['Electric Kettle', 'LED Desk Lamp', 'Car Phone Mount', 'Camera', 'Tea'],
        ['White Noise Machine', 'Hiking Boots', 'Backpack', 'Noise-Canceling Headphones'],
        ['Laptop', 'Smartwatch', 'Backpack', 'Noise-Canceling Headphones'],
        ['Coffee Maker', 'Wireless Earbuds', 'Fitness Tracker', 'Tea', 'Coffee', 'Espresso Machine'],
        ['Running Shoes', 'Yoga Mat', 'Fitness Resistance Bands', 'Water Bottle', 'Fitness Tracker'],
        ['Drone', 'Camera', 'External Hard Drive', 'Gaming Mouse', 'Laptop'],
        ['Noise-Canceling Headphones', 'Smart Thermostat', 'External Hard Drive', 'Laptop'],
        ['Wireless Router', 'Smartwatch'],
        ['Fitness Tracker', 'Camera', 'Travel Luggage', 'Yoga Mat', 'Water Bottle'],
        ['Espresso Machine', 'Bluetooth Keyboard', 'Backpack', 'Coffee', 'Tea'],
        ['Camera', 'Travel Luggage'],
        ['Coffee', 'Projector', 'Weighted Blanket', 'Coffee Maker'],
        ['Smart Light Bulbs', 'Weighted Blanket'],
        ['Tea'],
        ['Electric Kettle', 'Tea', 'Coffee'],
        ['Gaming Mouse', 'Virtual Reality Headset', 'Noise-Canceling Headphones'],
        ['Laptop', 'Running Shoes', 'Wireless Earbuds', 'Smartwatch', 'Yoga Mat', 'Fitness Tracker'],
        ['Blender', 'Portable Speaker', 'Backpack', 'Fitness Tracker', 'Coffee', 'Tea', 'Espresso Machine'],
        ['Gaming Mouse', 'Drone', 'Camera', 'Noise-Canceling Headphones'],
        ['Noise-Canceling Headphones', 'Smart Thermostat', 'External Hard Drive', 'Gaming Mouse', 'Laptop']
        
    ],
    "give_card": ["yes"] * 12 + ["no"] * 13
})

filename = "amazon_credit_card"


def main():
    
    print(f"\n --- Amazon User Data --- \n{amazon_user_df}")
    amazon_user_df.to_csv(f'./Final_Exam/{filename}.csv', index=False)
    
    print("\n\n --- Making basket Data --- \n\n")
    basket_product(amazon_user_df, "Products")

    print("\n\n ---------- Selecting Train and Test Data ---------- \n")
    amazon_user_dict = setup_train_test_data(amazon_user_df[["Credit_Score", "Income_Level", "Length_of_Credit_History", "give_card"]], label_col="give_card")
    
    print("\n\n ---------- Decision Tree ---------- \n")
    decision_tree_analysis(amazon_user_dict, filename) #, max_depth=20)
    
    print("\n\n ---------- Naive Bayes ---------- \n")
    amazon_user_dict_pred = multinomial_naive_bayes(amazon_user_dict, filename)
    
    print("\n\n ---------- SVM  ---------- \n")
    run_svm(amazon_user_dict, filename)
    
    print("DONE! Woo!")
    
    

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

def basket_product(df, column_of_interest):
    
    nested_dict = {}
    df2 = pd.DataFrame()

    for index, products_list in enumerate(df[column_of_interest]):
        nested_dict[index] = products_list

    unique_products = amazon_user_df['Products'].explode().unique().tolist()

    for index, products_list in nested_dict.items():
        for col in unique_products:
            col = col.replace(" ", "_")
            df2.loc[index, col] = ""
            for product in products_list:
                product = product.replace(" ", "_")
                if product == col:
                    df2.loc[index, col] = col
                    break
                else:
                    pass
    
    df2.to_csv(f'./Final_Exam/products_basket.csv', index=False)
    
    
def decision_tree_analysis(sample_dict, filename, criterion='gini', min_samples_split=2, min_samples_leaf=1, max_depth=None):
    
    train_labels = sample_dict["train_labels"]
    train_df = sample_dict["train_df"]
    test_labels = sample_dict['test_labels']
    test_df = sample_dict['test_df']
    
    
    MyDT=DecisionTreeClassifier(criterion=criterion, ##"entropy" or "gini"
                                splitter='best', 
                                max_depth=max_depth, 
                                min_samples_split=min_samples_split, 
                                min_samples_leaf=min_samples_leaf, 
                                min_weight_fraction_leaf=0.0, 
                                max_features=None, 
                                random_state=None, 
                                max_leaf_nodes=None, 
                                min_impurity_decrease=0.0, 
                                class_weight=None)    
    
    ## perform DT
    MyDT.fit(train_df, train_labels)
    ## plot the tree
    tree.plot_tree(MyDT)
    # plt.savefig(f"./CreatedVisuals/DecisionTree/{filename}_tree.png")
    # plt.close()
    train_feature_names=train_df.columns
    dot_data = tree.export_graphviz(MyDT, out_file=None,
                                    feature_names=train_feature_names, 
                                    filled=True, rounded=True,  
                                    special_characters=True)                                    
    graph = graphviz.Source(dot_data) 
    graph.format = 'png'
    graph.render(f"./Final_Exam/Part3_Visuals/{filename}_graphviz_tree") 
    
    ## Show the predictions from the DT on the test set
    print(f"\nActual for {filename}\n")
    print(test_labels)
    print("Prediction\n")
    DT_pred=MyDT.predict(test_df)
    print(DT_pred)
    
    score = accuracy_score(test_labels, DT_pred)
    print(f"\nThe accurary is for {filename} is : {score}\n")
    
    ## Show the confusion matrix
    cm = confusion_matrix(test_labels, DT_pred)
    print(f"\nThe confusion matrix for {filename} is:")
    print(cm)
    

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=MyDT.classes_)
    plt.figure(figsize=(18, 15))
    disp.plot(cmap='magma')
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Decision Tree Confusion Matrix\n - {filename} - \n{score}") 
    plt.tight_layout()
    plt.savefig(f"./Final_Exam/Part3_Visuals/{filename}_decisiontree_cm.png", dpi=300)
    plt.close()
    

def multinomial_naive_bayes(sample_dict, filename):
    
    train_labels = sample_dict["train_labels"]
    train_df = sample_dict["train_df"]
    test_labels = sample_dict['test_labels']
    test_df = sample_dict['test_df']

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
    plt.title(f"NB Confusion Matrix\n - {filename} - \n{score}") 
    plt.tight_layout()
    plt.savefig(f"./Final_Exam/Part3_Visuals/{filename}_NB_cm.png", dpi=300)
    plt.close()
    
    sample_dict["naive_pred"] = naive_pred
    
    return sample_dict

def run_svm(sample_dict, filename, visual_folder="./Final_Exam/Part3_Visuals"):
    
    ## Data
    train_labels = sample_dict["train_labels"]
    train_df = sample_dict["train_df"]
    test_labels = sample_dict['test_labels']
    test_df = sample_dict['test_df']
    

    SVM_Model1=LinearSVC(C=50, dual="auto")
    SVM_Model1.fit(train_df, train_labels)

    # print("SVM 1 prediction:\n", SVM_Model1.predict(test_df))
    # print("Actual:")
    # print(test_labels)
    
    score = accuracy_score(test_labels, SVM_Model1.predict(test_df))
    print(f"\nThe accurary is for Linear SVM {filename} is : {score}\n")

    SVM_matrix = confusion_matrix(test_labels, SVM_Model1.predict(test_df))
    # print("\nThe confusion matrix for Linear SVM is:")
    # print(SVM_matrix)
    # print("\n\n")
    
    print(f"Making linear visual for {filename}")
    disp = ConfusionMatrixDisplay(confusion_matrix=SVM_matrix, display_labels=SVM_Model1.classes_)
    plt.figure(figsize=(18, 15))
    disp.plot(cmap='magma')
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Confusion Matrix Linear SVM\n- {filename} -\n{score}") 
    plt.tight_layout()
    plt.savefig(f"{visual_folder}/{filename}_linear_svm_cm.png", dpi=300)
    plt.close()
    
    
    #--------------other kernels
    ## RBF
    print("--- Starting RBF ---")
    SVM_Model2=sklearn.svm.SVC(C=1.0, kernel='rbf', gamma="auto")
    SVM_Model2.fit(train_df, train_labels)

    # print("SVM prediction:\n", SVM_Model2.predict(test_df))
    # print("Actual:")
    # print(test_labels)
    
    score = accuracy_score(test_labels, SVM_Model2.predict(test_df))
    print(f"\nThe accurary is for rbf SVM {filename} is : {score}\n")

    
    SVM_matrix = confusion_matrix(test_labels, SVM_Model2.predict(test_df))
    # print("\nThe confusion matrix for rbf SVM is:")
    # print(SVM_matrix)
    # print("\n\n")
    
    print(f"Making rbf visual for {filename}")
    disp = ConfusionMatrixDisplay(confusion_matrix=SVM_matrix, display_labels=SVM_Model1.classes_)
    plt.figure(figsize=(18, 15))
    disp.plot(cmap='magma')
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Confusion Matrix RBF SVM\n- {filename} -\n{score}") 
    plt.tight_layout()
    plt.savefig(f"{visual_folder}/{filename}_rbf_svm_cm.png", dpi=300)
    plt.close()


    for d in range(1, 5): # [1, 5, 10, 20, 50]:
        ## POLY
        print("--- Starting Poly ---")
        SVM_Model3=sklearn.svm.SVC(C=1, kernel='poly', degree=d, gamma="scale")
        SVM_Model3.fit(train_df, train_labels)

        score = accuracy_score(test_labels, SVM_Model3.predict(test_df))
        print(f"\nThe accurary is for poly {d} SVM {filename} is : {score}\n")

        SVM_matrix = confusion_matrix(test_labels, SVM_Model3.predict(test_df))
        
        print(f"Making poly visual for {filename} - {d}")
        disp = ConfusionMatrixDisplay(confusion_matrix=SVM_matrix, display_labels=SVM_Model1.classes_)
        plt.figure(figsize=(18, 15))
        disp.plot(cmap='magma')
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Confusion Matrix Poly SVM - {d}\n - {filename} -\n{score}") 
        plt.tight_layout()
        plt.savefig(f"{visual_folder}/{filename}_poly_svm_cm_{d}.png", dpi=300)
        plt.close()


# DO NOT REMOVE!!!
if __name__ == "__main__":
    main()
