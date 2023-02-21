# topic-extraction 

Clone the repository with: git clone https://github.com/av-ll/topic-extraction

Change directory into the repository folder:

cd topic-extraction

create environment with conda:

conda env create -f environment.yml

activate environment:

conda activate iu-data-analysis

Now you can inspect the code and run it.

Start with load.py which extracts the data and filters it into a csv file

Then you can inspect clean.py which cleans and lemmatizes the data

You can run wordcloud_vis.py which creates a wordcloud visualization to a png file and discusses the results

Finally you can inspect both bow_tfidf_LDA and bow_tfidf_LSA which

contain the 2 topic modelling techniques described in the conception phase



