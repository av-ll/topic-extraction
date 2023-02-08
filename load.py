# Import libraries
import pandas as pd
import json
import numpy as np
import py7zr

# extract data from 7z file
archive = py7zr.SevenZipFile('raw_data.7z', mode='r')
archive.extractall()
archive.close()


#read json file
def data_retrieval():
    with open("raw_data.json") as f:
        file = json.load(f)
        return file

data = data_retrieval()

# We can see that we have 36 keys each corresponding to a specific area of the municipality/city
# Our objective is not check individual parts of the municipality but to check what topics are being discussed as a whole

len(list(data.keys())[0]), list(data.keys())[0]

# Storing all the keys in a list
data_keys = list(data.keys())

print(data[data_keys[1]])

#We can see some useful parameters

#body - the text we intend to process using nlp
#created - the timestamp of the complaint can be used to check 
#the changes of complaint topic over time but it is out of scope of this assignment
#rating - we can filter for bad ratings
#It should be noted that the analysis per part of the city and over-time 
#is out of the scope of this assignment and will not be explored further

def return_parameters():
    body = []
    rating = []
    for key in data_keys:
        x = data[key]
        for value in x :
            body.append(value['body'])
            rating.append(value['rating'])
    final_dataframe = pd.DataFrame([body,rating]).transpose()
    final_dataframe.columns=['body','rating']
    return final_dataframe
        
dataframe = return_parameters()

print(dataframe['body'][2])

#We can see that some positive reviews are also present but
#we are just looking for complaints so we will filter by rating below 2 to remove
#positive feedback

dataframe = dataframe[dataframe['rating'] < 3]

dataframe = dataframe.drop(['rating'],axis=1)

len(dataframe)

#We can see that we have around 21000 data points, 
#we saved the data to csv

dataframe.to_csv('data.csv',index=False)


