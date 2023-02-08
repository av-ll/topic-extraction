import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


tqdm.pandas()

data = pd.read_csv('clean_data.csv')

data = data.squeeze()

# Initialize TfidfVectorizer

tfidf =  TfidfVectorizer(max_df=.10,min_df=1/1000)

# Apply TF-IDF to the text data

tfidf_data = tfidf.fit_transform(data.apply(lambda x: np.str_(x)))

# Initialize LatentDirichletAllocation with tf-idf

from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=8, random_state=0, learning_method='batch')

xtopics = lda.fit_transform(tfidf_data)

# Get the most important terms for each topic
# We choose 5 topics as educated guess through try and fail hyperparameter tuning

n_top_words = 5
feature_names = tfidf.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
  print("Topic %d:" % (topic_idx + 1))
  print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

# We can see from the 5 extracted topics that likely the citizens 
# are most unhappy with :
# Topic 1 - Lack of Housing and tourist attractions
# Topic 2 - Not enough grocery stores, gas stations, businesses
# Topic 3 - Weather this is something the municipality cannot do anything about,
# public transportation in winter is also a concern possibly related
# Topic 4 - life can get boring 
# Topic 5 - More options for fast food and restaurants
# Topic 6 - Higher standards of community and family life required
# Topic 7 - Lack of safety in the neighborhoods at night, needs more policing
# Topic 8 - More opportunities to work and better wages

top_topic = xtopics.argmax(axis=1)

# count the number of documents assigned to each topic
topic_counts = np.bincount(top_topic)

print("\nNumber of documents assigned to each topic:", topic_counts)

# We can see the topics seem well separated so 8 is a reasonable amount to be considered
# If we consider the topic frequency We can see that 3 most frequently discussed topics are :
# Topic 6, 5764, Topic 4, 4156, and Topic 8, 2368 
# So perhaps these topics should be addressed first

# Initialize bow

bow = CountVectorizer(ngram_range=(1,2),stop_words='english',max_df=0.10,min_df=1/1000)

# fit it to the clean data
bow_data = bow.fit_transform(data.apply(lambda x: np.str_(x)))

lda1 = LatentDirichletAllocation(n_components=10, random_state=0, learning_method='batch')
ytopics = lda1.fit_transform(bow_data)

# Get the most important terms for each topic
# We choose 5 topics as educated guess through try and fail hyperparameter tuning

n_top_words = 5
feature_names1 = bow.get_feature_names_out()
for topic_idx, topic in enumerate(lda1.components_):
  print("Topic %d:" % (topic_idx + 1))
  print(" ".join([feature_names1[i]
  for i in topic.argsort()
    [:-n_top_words - 1:-1]]))




# We can see from the 5 extracted topics that likely the citizens 
# are most unhappy with :
# Topic 1 - More Work opportunities nearby
# Topic 2 - high crime rate possibly in parks and on the road
# Topic 3 - Dangerous walking at night in the street
# Topic 4 - Want nice looking and safe neighborhoods
# Topic 5 - More police and law enforcement on the roads
# Topic 6 - More opportunity for students, too small town and little change
# Topic 7 - Concern with raising a family 
# Topic 8 - Lack of restaurants and fast food stores
# Topic 9 - Low pay and high housing costs
# Topic 10 - Apparently horribly racist white kids

top_topic1 = ytopics.argmax(axis=1)

topic_counts1 = np.bincount(top_topic1)

print("\nNumber of documents assigned to each topic:", topic_counts1)

# We can see the topics seem well separated so 10 is a reasonable amount to be considered
# If we consider the topic frequency We can see that 3 most frequently discussed topics are :
# Topic 6, 2671 , Topic 1, 2386, and Topic 5, 2283
# So perhaps these topics should be addressed first

