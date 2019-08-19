import pickle
import numpy as np
import pandas as pd
import gensim as gs
import string

data_imdb = pd.read_csv('emnlp-2015-data/imdb-train.txt.ss', sep="    ", header=None, engine ='python',names=["full_review"])
data_yelp = pd.read_csv('emnlp-2015-data/yelp-2015-train.txt.ss', sep="    ", header=None, engine ='python',names=["full_review"])
imdb_sample = data_imdb.sample(frac = 0.3, replace = False)
yelp_sample = data_yelp.sample(frac = 0.3, replace = False)
del data_imdb
del data_yelp

data = pd.concat([imdb_sample,yelp_sample], axis = 0)

#make lists with ratings and sentcences
data["split"] = data['full_review'].str.split("\t\t")
ratings_list = [ls[2] for ls in data["split"]] #not needed for corpus but may be usefull for later
text_list = [ls[3] for ls in data["split"]]
sentences_raw = [sent for txt in text_list for sent in txt.split("<sssss>")]
sentences_list = [0]*len(sentences_raw)

#preprocess: all lower case, no accents, no punctuation, split into list of words
for ii,sent in enumerate(sentences_raw):
    sentences_list[ii] = sent.translate(str.maketrans('', '', string.punctuation))
    sentences_list[ii] = gs.utils.simple_preprocess(sent,deacc=True)

with open("corpus_train","wb") as outfile:
    pickle.dump(sentences_list,outfile)

model = gs.models.Word2Vec(sentences_list,size=200, window=5)
model.save("w2v_model")
