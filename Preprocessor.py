import os
import pickle
import string

import gensim as gs
import numpy as np
import pandas as pd

from typing import Tuple, List, Dict, Union

from DocSenTypes import *


class Preprocessor:

    def __init__(self, w2v_path: str="data/Word2Vec/", prep_path: str="data/Preprocessed/"):
        self._w2v_path = w2v_path
        self._prep_path = prep_path

        self._imdb_rating_index = 2
        self._imdb_review_index = 3

        self._unknown_word_key = 'UNK'

    def preprocess(self, path: str, name: str, w2v_model_name: str, persist: bool = True)\
            -> Tuple[List[TDocumentInd], List[TRating], TEmbedding]:
        """
        Preprocess IMDB data: Extract text and rating data and replace words by vocabulary ids.
        :param path: Path to the data file to process
        :param name: Name of the data to process
        :param w2v_model_name: Name of the word2vec model to use for embedding vocabulary encoding
        :param persist: If True, results of the preprocessing will be persisted
        (review text data, vocabulary indexed documents, ratings and embedding matrix)
        :return: List of documents with vocabulary indices instead of words, list of ratings and word embedding matrix
        """
        # load data and extract text information as X and rating as y:
        data = pd.read_csv(path, sep="    ", header=None, engine='python', names=["full_review"])
        data["split"] = data['full_review'].str.split("\t\t")

        # y is a list of gold star ratings for reviews
        y_data = [ls[self._imdb_rating_index] for ls in data["split"]]

        # X is a list with all documents, where documents are lists of sentences and each sentence-list
        # contains single words as strings
        X_data_text = [ls[self._imdb_review_index] for ls in data["split"]]

        # Separate and preprocess words in sentences
        for ii, doc in enumerate(X_data_text):
            X_data_text[ii] = doc.split("<sssss>")
            for jj, sent in enumerate(X_data_text[ii]):
                X_data_text[ii][jj] = sent.translate(str.maketrans('', '', string.punctuation))
                X_data_text[ii][jj] = gs.utils.simple_preprocess(sent, min_len=1, max_len=20, deacc=True)

        embedding, word2index = self.get_embedding(w2v_model_name)

        X_data_index = self._words_to_vocab_index(X_data_text, word2index)

        if persist:
            embedding_path = os.path.join(self._prep_path, f"{w2v_model_name}_embedding_matrix")
            if not os.path.isfile(embedding_path):
                np.save(embedding_path, embedding)
            with open(os.path.join(self._prep_path, f"X_{name}_text"), "wb") as savefile:
                pickle.dump(X_data_text, savefile)
            with open(os.path.join(self._prep_path, f"X_{name}"), "wb") as savefile:
                pickle.dump(X_data_index, savefile)
            with open(os.path.join(self._prep_path, f"y_{name}"), "wb") as savefile:
                pickle.dump(y_data, savefile)

        return X_data_index, y_data, embedding

    def _words_to_vocab_index(self, documents: List[TDocumentStr],
                              word2index: Dict[str, int]) -> List[TDocumentInd]:
        """
        Replace each word in the training data by it´s index in the vocab
        :param documents: List of documents containing lists of sentences containing lists of words
        :param word2index: Dict word -> vocab index
        :return: List of documents containing lists of sentences containing lists of vocabulary ids
        """
        documents_ind = []
        for document in documents:
            document_ind = []
            documents_ind.append(document_ind)
            for sentence in document:
                sentence_ind = []
                document_ind.append(sentence_ind)
                for word in sentence:
                    if word in word2index:
                        sentence_ind.append(word2index[word])
                    else:
                        sentence_ind.append(word2index[self._unknown_word_key])
        return documents_ind

    def get_embedding(self, w2v_model_name: str) -> Tuple[TEmbedding, Dict[TWord, TVocabIndex]]:
        """
        Load word embedding from the given word2vec model and extend it with vectors for unknown words and padding.
        :param w2v_model_name: Name of the word2vec model to use
        :return: Word embedding matrix and word2index mapping
        """
        w2v_model_path = os.path.join(self._w2v_path, f"{w2v_model_name}_w2v_model")
        w2v_word_vectors_path = os.path.join(self._w2v_path, f"{w2v_model_name}_w2v_model.wv.vectors.npy")
        if not (os.path.isfile(w2v_model_path) and os.path.isfile(w2v_word_vectors_path)):
            raise FileNotFoundError(f"Can't find a Word2Vec model with name '{w2v_model_name}' on path '{self._w2v_path}'")

        model = gs.models.KeyedVectors.load(w2v_model_path)
        wv = model.wv
        del model
        # embedding matrix is orderd by indices in model.wv.voacab
        word2index = {token: token_index for token_index, token in enumerate(wv.index2word)}
        # embedding = np.load(w2v_word_vectors_path)
        embedding = wv.vectors
        unknown_vector = np.mean(embedding, axis=0)
        padding_vector = np.zeros(len(embedding[0]))
        embedding = np.append(embedding, unknown_vector.reshape((1, -1)), axis=0)
        embedding = np.append(embedding, padding_vector.reshape((1, -1)), axis=0)
        word2index[self._unknown_word_key] = len(embedding) - 2  # UNK = unknown words, map to vector we just appended
        return embedding, word2index

    def word2vec(self, paths: Union[str, List[str]], name: str, dim: int = 200, overwrite: bool = True, sample_frac: float = 0.3) \
            -> Tuple[TEmbedding, Dict[TWord, TVocabIndex]]:
        w2v_model_path = os.path.join(self._w2v_path, f'{name}_w2v_model')
        w2v_corpus_path = os.path.join(self._w2v_path, f'{name}_w2v_corpus_train')
        if overwrite or (not os.path.isfile(w2v_model_path)):
            data = pd.DataFrame()
            if type(paths) == List[str]:
                for path in paths:
                    data_in = pd.read_csv(path, sep="    ", header=None, engine='python', names=["full_review"])
                    sample = data_in.sample(frac=sample_frac, replace=False)
                    data = pd.concat([data, sample], axis=0)
                    del data_in
            else:
                data_in = pd.read_csv(paths, sep="    ", header=None, engine='python', names=["full_review"])
                data = data_in.sample(frac=sample_frac, replace=False)

            # make lists with ratings and sentcences
            data["split"] = data['full_review'].str.split("\t\t")
            ratings_list = [ls[2] for ls in data["split"]]  # not needed for corpus but may be usefull for later
            text_list = [ls[3] for ls in data["split"]]
            sentences_raw = [sent for txt in text_list for sent in txt.split("<sssss>")]
            sentences_list = [0] * len(sentences_raw)

            # preprocess: all lower case, no accents, no punctuation, split into list of words
            for ii, sent in enumerate(sentences_raw):
                sentences_list[ii] = sent.translate(str.maketrans('', '', string.punctuation))
                sentences_list[ii] = gs.utils.simple_preprocess(sent, deacc=True)

            with open(w2v_corpus_path, "wb") as outfile:
                pickle.dump(sentences_list, outfile)

            model = gs.models.Word2Vec(sentences_list, size=dim, window=5)
            model.save(w2v_model_path)

        return self.get_embedding(name)


if __name__ == '__main__':
    p = Preprocessor()

    p.word2vec(['emnlp-2015-data/imdb-train.txt.ss', 'emnlp-2015-data/yelp-2015-train.txt.ss'], 'imdb+yelp', overwrite=False)

    file_suffix_length = len('.txt.ss') + 1
    for file in os.listdir('data/Dev'):
        p.preprocess(f'data/Dev/{file}', file[:file_suffix_length], 'imdb+yelp')
