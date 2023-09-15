'''
This is the API data gathering and cleaning for the News Api
This code follows the tutorial created by Dr. Amy Gates
- only going till the cloud graphic, not any of the models
'''

import requests  ## for getting data from a server GET
import re  
import pandas as pd    
from pandas import DataFrame

## To tokenize and vectorize text type data
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import sklearn


#############################################
##
## Contecting to News API
##
#############################################

# My API key for https://newsapi.org/
kat_api_key = "3755f6b8d041439cb65c0e078a55bca0"

## TOPICS
topics = ['wildfire', 'fire', 'drought', 'burn', 'weather']

## CREAT NEW CSV FOR HEADLINES
filename = "NewsHeadlines.csv"
MyFile = open(filename, "w")  # 'w' -> write to new

WriteThis = "LABEL,Date,Source,Title,Headline\n"
MyFile.write(WriteThis)
MyFile.close()

### GATHERING THE DATA 

## DEFINING THE ENDPOINT AND THE QUERY
endpoint = "https://newsapi.org/v2/everything"

for topic in topics:
    
    URLPost = {'apiKey' : kat_api_key,
               'q':topic,
               'language':'en'
    }
    
    response = requests.get(endpoint, URLPost)
    jsontxt = response.json()
    
    MyFILE=open(filename, "a") # "a" for append to add stuff
    LABEL=topic
    for items in jsontxt["articles"]:
        # print(items, "\n\n\n")
                  
        #Author=items["author"]
        #Author=str(Author)
        #Author=Author.replace(',', '')
        
        ## SOURCE
        Source=items["source"]["name"]
        # print(Source)
        
        ## DATE
        Date=items["publishedAt"]
        NewDate=Date.split("T")
        Date=NewDate[0]
        # print(Date)
        
        ## TITLE
        #  - Replace punctuation with space
        #  - Accept one or more copies of punctuation         
        #  - plus zero or more copies of a space
        #  - and replace it with a single space
        Title=items["title"]
        Title=str(Title)
        Title=re.sub(r'[,.;@#?!&$\-\']+', ' ', str(Title), flags=re.IGNORECASE)
        Title=re.sub(' +', ' ', str(Title), flags=re.IGNORECASE)
        Title=re.sub(r'\"', ' ', str(Title), flags=re.IGNORECASE)
        Title=re.sub(r'[^a-zA-Z]', " ", str(Title), flags=re.VERBOSE)
        Title=Title.replace(',', '')
        Title=' '.join(Title.split())
        Title=re.sub("\n|\r", "", Title)
        # print(Title)

        ## HEADLINE
        Headline=items["description"]
        Headline=str(Headline)
        Headline=re.sub(r'[,.;@#?!&$\-\']+', ' ', Headline, flags=re.IGNORECASE)
        Headline=re.sub(' +', ' ', Headline, flags=re.IGNORECASE)
        Headline=re.sub(r'\"', ' ', Headline, flags=re.IGNORECASE)
        Headline=re.sub(r'[^a-zA-Z]', " ", Headline, flags=re.VERBOSE)
        Headline=Headline.replace(',', '') # commas are bad for csv
        Headline=' '.join(Headline.split())
        Headline=re.sub("\n|\r", "", Headline)
        
        ### AS AN OPTION - remove words of a given length............
        ### Headline = ' '.join([wd for wd in Headline.split() if len(wd)>3])
        
        WriteThis=str(LABEL)+","+str(Date)+","+str(Source)+","+ str(Title) + "," + str(Headline) + "\n"
        # print(WriteThis)
        
        MyFILE.write(WriteThis)
        
    ## CLOSE THE FILE
    MyFILE.close()

#############################################
##
## Tokenize and Vectorize the Headlines
##
#############################################

BBC_DF=pd.read_csv(filename, error_bad_lines=False)    

## REMOVE any rows with NaN in them
BBC_DF = BBC_DF.dropna()

HeadlineLIST=[]
LabelLIST=[]
for nexthead, nextlabel in zip(BBC_DF["Headline"], BBC_DF["LABEL"]):
    HeadlineLIST.append(nexthead)
    LabelLIST.append(nextlabel)
    
    
NewHeadlineLIST=[]
for element in HeadlineLIST:
    AllWords=element.split(" ")
    
    ## Now remove words that are in your topics
    NewWordsList=[]
    for word in AllWords:
        word=word.lower()
        if word in topics:
            pass
        else:
            NewWordsList.append(word)
            
    ##turn back to string
    NewWords=" ".join(NewWordsList)
    NewHeadlineLIST.append(NewWords)
    
HeadlineLIST=NewHeadlineLIST

### Vectorize
## Instantiate your CV
MyCountV=CountVectorizer(
        input="content",  ## because we have a csv file
        lowercase=True, 
        stop_words = "english",
        max_features=50
        )

## Use your CV 
MyDTM = MyCountV.fit_transform(HeadlineLIST)  # create a sparse matrix

ColumnNames=MyCountV.get_feature_names_out()


## Build the data frame
MyDTM_DF=pd.DataFrame(MyDTM.toarray(),columns=ColumnNames)

## Convert the labels from list to df
Labels_DF = DataFrame(LabelLIST,columns=['LABEL'])

##Save original DF - without the lables
My_Orig_DF=MyDTM_DF

## Now - let's create a complete and labeled
## dataframe:
dfs = [Labels_DF, MyDTM_DF]

Final_News_DF_Labeled = pd.concat(dfs,axis=1, join='inner')
 

#############################################
##
## Word Cloud
##
#############################################


List_of_WC=[]

for mytopic in topics:

    tempdf = Final_News_DF_Labeled[Final_News_DF_Labeled['LABEL'] == mytopic]
    
    tempdf =tempdf.sum(axis=0,numeric_only=True)
    
    #Make var name
    NextVarName=str("wc"+str(mytopic))
    
    ###########
    ## Create and store in a list the wordcloud OBJECTS
    #########
    NextVarName = WordCloud(width=1000, height=600, background_color="white",
                   min_word_length=4, #mask=next_image,
                   max_words=200).generate_from_frequencies(tempdf)
    
    ## Here, this list holds all three wordclouds I am building
    List_of_WC.append(NextVarName)
    

##------------------------------------------------------------------
##########
########## Create the wordclouds
##########
fig=plt.figure(figsize=(25, 25))
NumTopics=len(topics)
for i in range(NumTopics):
    ax = fig.add_subplot(NumTopics,1,i+1)
    plt.imshow(List_of_WC[i], interpolation='bilinear')
    plt.axis("off")
    plt.savefig("NewClouds.pdf")