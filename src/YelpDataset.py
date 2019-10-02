import os
import pickle
import random
import re
import string
from typing import Tuple, Dict, Union

import gensim as gs
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from DocSenTypes import *
from Word2Vector import Word2Vector


class YelpDataset(Dataset):

    def __init__(self, data_paths: Union[str, List[str]], name: str, w2v_model_name: str = None,
                 overwrite: bool = False,
                 embedding_dim: int = 200, w2v_sample_frac: float = 0.3,
                 w2v_path: str = "../data/Word2Vec/", prep_path: str = "../data/Preprocessed/",
                 use_reduced_dataset: float = 0):
        """
        Load a given YELP rating dataset. If <overwrite> is true or there is no persisted data for the given <_name> yet,
        the data will be preprocessed and the results persisted under the given paths <w2v_path> and <prep_path>.

        :param data_paths: One or multiple paths to raw data files to load
        :param name: Name of the data to load (for naming output files)
        :param overwrite: If there are files with the given _name already, rebuild model and overwrite them or load them?
        :param w2v_path: Path to Word2Vec directory (for persistence)
        :param prep_path: Path to data preprocessing directory (for persistence)
        :param use_reduced_dataset: If > 0, use only those data with class=min(classes) and class=max(classes) and
        randomly sample a fraction of <use_reduced_dataset> of these data.
        """
        self._data_paths = data_paths
        self._prep_path = prep_path
        self._name = name
        if w2v_model_name:
            self._w2v_model_name = w2v_model_name
        else:
            self._w2v_model_name = name
        self._overwrite = overwrite

        self._embedding_dim = embedding_dim
        self._w2v_sample_frac = w2v_sample_frac
        self._w2v = Word2Vector(data_paths, w2v_path, self._w2v_model_name, self._overwrite, self._embedding_dim)

        self._yelp_rating_key = 'stars'
        self._yelp_review_key = 'text'

        self._X_data, self._y_data, self.embedding, self.word2index = self._load(use_reduced_dataset)
        self.index2word = {index: word for (word, index) in self.word2index.items()}
        self.classes = sorted([int(y) for y in set(self._y_data)])

        self.num_classes = len(self.classes)

        self.unknown_word_key = self._w2v.unknown_word_key
        self.padding_word_key = self._w2v.padding_word_key

    def get_class_distr(self, labels):
        class_distr = np.zeros((len(self.classes),))
        for y in labels:
            y = int(y)
            class_distr[y - 1] += 1
        return class_distr

    def __getitem__(self, index):
        """
        Get tuple of data and label, where the label is the index of the class in YelpDataset.classes.
        """
        return self._X_data[index], self.classes.index(int(self._y_data[index]))

    def __len__(self):
        return len(self._X_data)

    def _load(self, use_reduced_dataset) -> Tuple[
        List[TDocumentInd], List[TRating], TEmbedding, Dict[TWord, TVocabIndex]]:
        """
        Preprocess Yelp data: Extract text and rating data and replace words by vocabulary ids.
        :return: List of documents with vocabulary indices instead of words, list of ratings and word embedding matrix
        """
        if self._overwrite or \
                (not os.path.isfile(self._X_path())) or \
                (not os.path.isfile(self._y_path())):
            print("No persisted data found. Preprocessing data...")
            X_data, y_data, embedding, word2index = self._preprocess()
        else:
            print("Persisted data found. Loading...")
            X_data, y_data, embedding, word2index = self._load_preprocessed()

        if use_reduced_dataset > 0:
            classes = set(y_data)
            class_min = min([int(y) for y in classes])
            class_max = max([int(y) for y in classes])
            X_reduced = []
            y_reduced = []
            for i, X in enumerate(X_data):
                label = int(y_data[i])
                if (label == class_min or label == class_max) and (random.uniform(0, 1) < use_reduced_dataset):
                    X_reduced.append(X)
                    y_reduced.append(y_data[i])
            X_data = X_reduced
            y_data = y_reduced

        return X_data, y_data, embedding, word2index

    def _load_preprocessed(self) -> Tuple[List[TDocumentInd], List[TRating], TEmbedding, Dict[TWord, TVocabIndex]]:
        with open(self._X_path(), "rb") as file:
            X_data = pickle.load(file)
        with open(self._y_path(), "rb") as file:
            y_data = pickle.load(file)
        embedding, word2index = self._w2v.get_embedding(X_data)
        return X_data, y_data, embedding, word2index

    def _preprocess(self) -> Tuple[List[TDocumentInd], List[TRating], TEmbedding, Dict[TWord, TVocabIndex]]:
        # load data and extract text information as X and rating as y:
        data = pd.DataFrame()
        if type(self._data_paths) == List[str]:
            for path in self._data_paths:
                data_in = pd.read_json(path, lines=True)
                data = pd.concat([data, data_in], axis=0)
                del data_in
        else:
            data = pd.read_json(self._data_paths, lines=True)
        # y is a list of gold star ratings for reviews
        y_data = data[self._yelp_rating_key]
        # X is a list with all documents, where documents are lists of sentences and each sentence-list
        # contains single words as strings
        X_data_text = data[self._yelp_review_key]
        # Separate and preprocess words in sentences
        X_data_prep = []
        for i, doc in enumerate(X_data_text):
            if i % 1000 == 0:
                print(f"Processing documents {i} - {i+999} of {len(X_data_text)}...")
            X_data_prep.append([])
            split = re.split('\!|\.|\?|\;|\:|\n', doc)
            for sentence in split:
                sentence = sentence.translate(str.maketrans('', '', string.punctuation))
                X_data_prep[-1].append(gs.utils.simple_preprocess(sentence, min_len=1, max_len=20, deacc=True))
        print("Building word2vec model...")
        embedding, word2index = self._w2v.get_embedding(X_data_prep)
        X_data_index = self._words_to_vocab_index(X_data_prep, word2index)
        with open(self._X_text_path(), "wb") as savefile:
            pickle.dump(X_data_prep, savefile)
        with open(self._X_path(), "wb") as savefile:
            pickle.dump(X_data_index, savefile)
        with open(self._y_path(), "wb") as savefile:
            pickle.dump(y_data, savefile)
        return X_data_index, y_data, embedding, word2index

    def _words_to_vocab_index(self, documents: List[TDocumentStr], word2index: Dict[str, int]) -> List[TDocumentInd]:
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
                        sentence_ind.append(word2index[self._w2v.unknown_word_key])
        return documents_ind

    def _y_path(self) -> str:
        return os.path.join(self._prep_path, f"y_{self._name}")

    def _X_text_path(self) -> str:
        return os.path.join(self._prep_path, f"X_{self._name}_text")

    def _X_path(self) -> str:
        return os.path.join(self._prep_path, f"X_{self._name}")
