import os
import pickle
import string

import gensim as gs
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

w2v_path = "data/Word2Vec/"
prep_path = "data/Preprocessed/"


def preprocess(path, name):
    # load data and extract text information as X and rating as y:
    data = pd.read_csv(path, sep="    ", header=None, engine='python', names=["full_review"])
    data["split"] = data['full_review'].str.split("\t\t")

    # y is a list of gold star ratings for reviews
    y_data = [ls[2] for ls in data["split"]]

    with open(os.path.join(prep_path, f"y_{name}"), "wb") as savefile:
        pickle.dump(y_data, savefile)

    # X is a list with all documents, where documents are lists of sentences and each sentence-list
    # contains single words as strings
    X_data = [ls[3] for ls in data["split"]]
    for ii, doc in enumerate(X_data):
        X_data[ii] = doc.split("<sssss>")
        for jj, sent in enumerate(X_data[ii]):
            X_data[ii][jj] = sent.translate(str.maketrans('', '', string.punctuation))
            X_data[ii][jj] = gs.utils.simple_preprocess(sent, min_len=1, max_len=20, deacc=True)

    with open(os.path.join(prep_path, f"X_{name}_text"), "wb") as savefile:
        pickle.dump(X_data, savefile)

    model = gs.models.KeyedVectors.load(os.path.join(w2v_path, "w2v_model"))
    kv = model.wv
    del model
    # embedding matrix is orderd by inices in model.wv.voacab
    word2index = {token: token_index for token_index, token in enumerate(kv.index2word)}
    embedding = np.load(os.path.join(w2v_path, "w2v_model.wv.vectors.npy"))
    unknown = np.mean(embedding, axis=0)
    padding = np.zeros(len(embedding[0]))
    embedding = np.append(embedding, unknown)  # vector for unknown words
    embedding = np.append(embedding, padding)

    embedding_path = os.path.join(prep_path, "embedding_matrix")
    if not os.path.isfile(embedding_path):
        np.save(embedding_path, embedding)

    word2index["UNK"] = len(embedding) - 2  # UNK = unknown words, map to vector we just appended

    # replace each word in the training data by it´s index in the vocab
    X_data_index = []

    for document in X_data:
        document_ind = []
        X_data_index.append(document_ind)
        # Sentences
        for sentence in document:
            sentence_ind = []
            document_ind.append(sentence_ind)
            # Words
            for word in sentence:
                if word in word2index:
                    sentence_ind.append(word2index[word])
                else:
                    sentence_ind.append(word2index["UNK"])

    # pad sequences s.t. they all have the same length
    X_data_padded = []
    padding_index = len(embedding) - 1
    for doc in X_data_index:
        X_data_padded.append(pad_sequences(doc, maxlen=50, padding='post', truncating='post', value=padding_index))

    with open(os.path.join(prep_path, f"X_{name}"), "wb") as outfile:
        pickle.dump(X_data_padded, outfile)


# run on all the data
for file in os.listdir('data/Dev'):
    preprocess(f'data/Dev/{file}', file[:len('.txt.ss')])
