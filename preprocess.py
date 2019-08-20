import os
import pickle
import string

import gensim as gs
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences


def preprocess(path, name):
    # load data and extract text information as X and rating as y:
    data = pd.read_csv(path, sep="    ", header=None, engine='python', names=["full_review"])
    data["split"] = data['full_review'].str.split("\t\t")

    # y is a list of gold start ratings for reviews
    y_data = [ls[2] for ls in data["split"]]

    with open(f"y_{name}", "wb") as savefile:
        pickle.dump(y_data, savefile)

    # X is a list with all documents, where documents are lists of sentences and each sentence-list
    # contains single words as strings
    X_data = [ls[3] for ls in data["split"]]
    for ii, doc in enumerate(X_data):
        X_data[ii] = doc.split("<sssss>")
        for jj, sent in enumerate(X_data[ii]):
            X_data[ii][jj] = sent.translate(str.maketrans('', '', string.punctuation))
            X_data[ii][jj] = gs.utils.simple_preprocess(sent, deacc=True)

    with open(f"X_{name}_text", "wb") as savefile:
        pickle.dump(X_data, savefile)

    model = gs.models.KeyedVectors.load("w2v_model")
    kv = model.wv
    del model
    # embedding matrix is orderd by inices in model.wv.voacab
    word2index = {token: token_index for token_index, token in enumerate(kv.index2word)}
    embedding = np.load("w2v_model.wv.vectors.npy")
    unknown = np.mean(embedding, axis=0)
    padding = np.zeros(200)
    embedding = np.append(embedding, unknown)
    embedding = np.append(embedding, padding)
    word2index["UNK"] = len(embedding) - 2

    # replace each word in the training data by it´s index in the vocab
    X_data_index = []
    for ii in range(len(X_data)):
        X_data_index.append([])
        for jj in range(len(X_data[ii])):
            X_data_index[ii].append([])
            for kk in range(len(X_data[ii][jj])):
                if X_data[ii][jj][kk] in word2index:
                    X_data_index[ii][jj].append(word2index[X_data[ii][jj][kk]])
                else:
                    X_data_index[ii][jj].append(word2index["UNK"])

    # pad sequences s.t. they all have the same length
    X_data_padded = []
    for doc in X_data_index:
        X_data_padded.append(pad_sequences(doc, maxlen=50, padding='post', truncating='post', value=len(embedding) - 1))

    with open(f"X_{name}", "wb") as outfile:
        pickle.dump(X_data_padded, outfile)

    np.save("embedding_matrix", embedding)


# run on all the data
for file in os.listdir('emnlp-2015-data'):
    preprocess(f'emnlp-2015-data/{file}', file[:-7])
