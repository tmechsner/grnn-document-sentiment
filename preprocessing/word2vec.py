import pickle
import numpy as np
import pandas as pd
import gensim as gs

data = pd.read_csv('emnlp-2015-data/imdb-train.txt.ss', sep="    ", header=None, engine ='python',names=["full_review"])

#make lists with ratings and sentcences
data["split"] = data['full_review'].str.split("\t")
ratings_list = [ls[4] for ls in data["split"]] #not needed for corpus but may be usefull for later
text_list = [ls[6] for ls in data["split"]]
sentences_raw = [sent for txt in text_list for sent in txt.split("<sssss>")]
sentence_list = []

#preprocess: all lower case, no accents
for sent in sentences_raw:
    sentence_list.append(gs.utils.simple_preprocess(sent,deacc=True))

with open("imdb_corpus","wb") as outfile:
    pickle.dump(sentence_list,outfile)

with open("ratings_imdb_corpus","wb") as outfile:
    pickle.dump(ratings_list,outfile)

model = gs.models.Word2Vec(sentence_list,size=200, window=5)
model.save("w2v_model_imdb")