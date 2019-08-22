import os
import pickle
import string
from typing import Tuple, Dict, Union

import gensim as gs
import pandas as pd
from torch.utils.data import Dataset

from DocSenTypes import *
from Word2Vector import Word2Vector


class ImdbDataset(Dataset):

    def __init__(self, data_paths: Union[str, List[str]], name: str, w2v_model_name: str = None,
                 overwrite: bool = False,
                 embedding_dim: int = 200, w2v_sample_frac: float = 0.3,
                 w2v_path: str = "data/Word2Vec/", prep_path: str = "data/Preprocessed/"):
        """
        Load a given IMDB rating dataset. If <overwrite> is true or there is no persisted data for the given <_name> yet,
        the data will be preprocessed and the results persisted under the given paths <w2v_path> and <prep_path>.

        :param data_paths: One or multiple paths to raw data files to load
        :param name: Name of the data to load (for naming output files)
        :param overwrite: If there are files with the given _name already, rebuild model and overwrite them or load them?
        :param w2v_path: Path to Word2Vec directory (for persistence)
        :param prep_path: Path to data preprocessing directory (for persistence)
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
        self._w2v = Word2Vector(data_paths, w2v_path, self._w2v_model_name, self._overwrite, self._embedding_dim, self._w2v_sample_frac)

        self._imdb_rating_index = 2
        self._imdb_review_index = 3

        self._X_data, self._y_data, self.embedding, self.word2index = self._load()
        self.index2word = {index: word for (word, index) in self.word2index.items()}

    def __getitem__(self, index):
        return self._X_data[index], self._y_data[index]

    def __len__(self):
        return len(self._X_data)

    def _load(self) -> Tuple[List[TDocumentInd], List[TRating], TEmbedding, Dict[TWord, TVocabIndex]]:
        """
        Preprocess IMDB data: Extract text and rating data and replace words by vocabulary ids.
        :return: List of documents with vocabulary indices instead of words, list of ratings and word embedding matrix
        """
        if self._overwrite or \
                (not os.path.isfile(self._X_text_path())) or \
                (not os.path.isfile(self._X_path())) or \
                (not os.path.isfile(self._y_path())):
            print("No persisted data found. Preprocessing data...")
            X_data, y_data, embedding, word2index = self._preprocess()
        else:
            print("Persisted data found. Loading...")
            X_data, y_data, embedding, word2index = self._load_preprocessed()

        return X_data, y_data, embedding, word2index

    def _load_preprocessed(self) -> Tuple[List[TDocumentInd], List[TRating], TEmbedding, Dict[TWord, TVocabIndex]]:
        with open(self._X_path(), "rb") as file:
            X_data = pickle.load(file)
        with open(self._y_path(), "rb") as file:
            y_data = pickle.load(file)
        embedding, word2index = self._w2v.get_embedding()
        return X_data, y_data, embedding, word2index

    def _preprocess(self) -> Tuple[List[TDocumentInd], List[TRating], TEmbedding, Dict[TWord, TVocabIndex]]:
        # load data and extract text information as X and rating as y:
        data = pd.DataFrame()
        if type(self._data_paths) == List[str]:
            for path in self._data_paths:
                data_in = pd.read_csv(path, sep="    ", header=None, engine='python', names=["full_review"])
                data = pd.concat([data, data_in], axis=0)
                del data_in
        else:
            data = pd.read_csv(self._data_paths, sep="    ", header=None, engine='python', names=["full_review"])
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
        embedding, word2index = self._w2v.get_embedding()
        X_data_index = self._words_to_vocab_index(X_data_text, word2index)
        with open(self._X_text_path(), "wb") as savefile:
            pickle.dump(X_data_text, savefile)
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


if __name__ == '__main__':
    p = ImdbDataset('data/Dev/imdb-dev.txt.ss', 'imdb-dev', w2v_sample_frac=0.9)
    print(p[0])
    print(' '.join(list(map(lambda index: p.index2word[index], p[0][0][0]))))
    pass
