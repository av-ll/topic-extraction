import wordcloud
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('clean_data.csv')

comment_words = ''

stopwords = set()

for x in data.values:
     
    x = str(x)
    tokens = x.split()
     
    comment_words += " ".join(tokens)+" "
 
wordcloud1 = wordcloud.WordCloud(width = 900, height = 900,
                background_color ='black',
                min_font_size = 10).generate(comment_words)

                     
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud1)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()

# We can see that some words are too general like city, town, know, year,
# neighborhood. We should have this in mind when performing the vectorization
# and set some limits of frequency for TF-IDF and BOW. For example limiting the
# frequency of words to 10% which seems like a reasonable amount.

common_words =('lot','great','go','know','time','come','thing','place','good','like','live',
               'area','town','city','neighborhood','people','year','house'
               ,'community','well')

for x in common_words:
    stopwords.add(x)    

wordcloud2 = wordcloud.WordCloud(width = 900, height = 900,
                background_color ='black', stopwords=common_words,max_font_size=200,
                min_font_size = 10,random_state=1).generate(comment_words)
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud2)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.savefig('wordcloud.png')
plt.show()

# After removing some of this words from the wordcloud we can have a somewhat
# clearer perspective on some of recurrent words. From the image generate (wordcloud.png)
# for example we can identify some words like police, job, word, small, crime, that could
# constitute some frequent worries pertinent to the citizens 
