# import required libraries

import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

tqdm.pandas()

data = pd.read_csv('clean_data.csv')

# Turns pandas dataframe into series

data = data.squeeze()

# Initialize TfidfVectorizer with minimum document frequency

tfidf1 =  TfidfVectorizer(max_df=1/20)

# Apply TF-IDF to the text data

tfidf_data1 = tfidf1.fit_transform(data.apply(lambda x: np.str_(x)))

# Initialize Bag of Words N_Grams with minimum document frequency

bow1 = CountVectorizer(max_df=0.1,stop_words='english',ngram_range=(1,1),max_features=15000)

# Apply Bag of Words N_Grams to the text data

bow1_data = bow1.fit_transform(data.apply(lambda x: np.str_(x)))

# import LSA sklearn library

from sklearn.decomposition import TruncatedSVD

# We create a function to retrieve topics top 5 words
def get_top_words(components, feature_names, n=6):
    topics = []
    
    # Retrieve the indices of the top n words for each component
    for i, component in enumerate(components):
        
        # Convert the indices to actual words using the feature names
        top_words = [feature_names[x] for x in component.argsort()[:-n - 1:-1]]
        
        # Store the top words for each component as a topic
        topics.append('Topic {}: {}'.format(i+1, ', '.join(top_words)))
    return topics

lsa = TruncatedSVD(n_components=8,random_state=0)

lsa_result = lsa.fit_transform(tfidf_data1)

lsa_components = lsa.components_

feature_names = tfidf1.get_feature_names_out()

# Find the index of the highest component for each document
top_topic = lsa_result.argmax(axis=1)

# count the number of documents assigned to each topic
topic_counts = np.bincount(top_topic)

# Retrieve the top words for each topic
top_words = get_top_words(lsa_components, feature_names, n=6)
print('\nTF-IDF-LSA Topics:\n')
for i,topic in enumerate(top_words):
    print('\n'+topic+'\n'+"\nNumber of documents assigned to topic: "+str(topic_counts[i]),'\n')
    

# Some prevalent topics of discussion are identified
# We will discuss the Top 3
# Topic 1 - Likely relates to low pay and lack of options relating to night bars
# Topic 3 - Worries about minimum wage, employment available and pay
# Topic 2 - lack of places to eat, bars and nightlife



lsa1 = TruncatedSVD(n_components=15,random_state=0)

lsa_result1 = lsa1.fit_transform(bow1_data)

lsa_components1 = lsa1.components_

feature_names1 = bow1.get_feature_names_out()

top_topic1 = lsa_result1.argmax(axis=1)

# count the number of documents assigned to each topic
topic_counts1 = np.bincount(top_topic1)

top_words1 = get_top_words(lsa_components1, feature_names1, n=6)
print('BOW-LSA Topics:\n')
for i,topic in enumerate(top_words1):
    print('\n'+topic+'\n'+"\nNumber of documents assigned to topic: "+str(topic_counts1[i]),'\n')
    
    
# Some prevalent topics of discussion are identified
# We will discuss the Top 3
# Topic 1 - Worries about family/community and work
# Topic 2 - Worries about lack of restaurants food, stores, businesses
# Topic 7 - Need for more police departments in the community 


