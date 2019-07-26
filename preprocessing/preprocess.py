import numpy as np
import gensim as gs
import pandas as pd
import string
import pickle

#load data and extract text information as X and rating as y:
imdb_train = pd.read_csv('emnlp-2015-data/imdb-train.txt.ss', sep="    ", header=None, engine ='python',names=["full_review"])
imdb_test = pd.read_csv('emnlp-2015-data/imdb-test.txt.ss', sep="    ", header=None, engine ='python',names=["full_review"])

imdb_train["split"] = imdb_train['full_review'].str.split("\t\t")
imdb_test["split"] = imdb_test['full_review'].str.split("\t\t")

#y is a list of gold start ratings for reviews
y_train = [ls[2] for ls in imdb_train["split"]]
y_test = [ls[2] for ls in imdb_test["split"]]

with open("y_train", "wb") as savefile:
    pickle.dump(y_train, savefile)

with open("y_test", "wb") as savefile:
   pickle.dump(y_test, savefile)

#X is a list with all documents, where documents are lists of sentences and each sentence-list
#contains single words as strings
X_train = [ls[3] for ls in imdb_train["split"]]
for ii,doc in enumerate(X_train):
    X_train[ii] = doc.split("<sssss>")
    for jj,sent in enumerate(X_train[ii]):
        X_train[ii][jj] = sent.translate(str.maketrans('', '', string.punctuation))
        X_train[ii][jj] = gs.utils.simple_preprocess(sent,deacc=True)

with open("X_train_text", "wb") as savefile:
    pickle.dump(X_train, savefile)

X_test = [ls[3] for ls in imdb_test["split"]]
for ii,doc in enumerate(X_test):
    X_test[ii] = doc.split("<sssss>")
    for jj,sent in enumerate(X_test[ii]):
        X_test[ii][jj] = sent.translate(str.maketrans('', '', string.punctuation))
        X_test[ii][jj] = gs.utils.simple_preprocess(sent,deacc=True)

with open("X_test_text", "wb") as savefile:
    pickle.dump(X_test, savefile)   

model = gs.models.KeyedVectors.load("w2v_model_imdb")
kv = model.wv
del model
#embedding matrix is orderd by inices in model.wv.voacab
word2index = {token: token_index for token_index, token in enumerate(kv.index2word)}
embedding = np.load("w2v_model_imdb.wv.vectors.npy")
unknown = np.mean(embedding,axis = 0)
embedding = np.append(embedding,unknown)
word2index["UNK"] = unknown

#replace each word in the training data by it´s index in the vocab
X_train_index = []
for ii in range(len(X_train)):
    X_train_index.append([])
    for jj in range(len(X_train[ii])):
        X_train_index[ii].append([])
        for kk in range(len(X_train[ii][jj])):
            if X_train[ii][jj][kk] in word2index:
                X_train_index[ii][jj].append(word2index[X_train[ii][jj][kk]])
            else:
                X_train_index[ii][jj].append(word2index["UNK"])

X_test_index = []
for ii in range(len(X_test)):
    X_test_index.append([])
    for jj in range(len(X_test[ii])):
        X_test_index[ii].append([])
        for kk in range(len(X_test[ii][jj])):
            if X_test[ii][jj][kk] in word2index:
                X_test_index[ii][jj].append(word2index[X_test[ii][jj][kk]])
            else:
                X_test_index[ii][jj].append(word2index["UNK"])

with open("X_train","wb") as outfile:
    pickle.dump(X_train_index, outfile)

with open("X_test","wb") as outfile:
    pickle.dump(X_test_index, outfile)

with open("embedding_matrix", "wb") as outfile:
    pickle.dump(embedding, outfile)
