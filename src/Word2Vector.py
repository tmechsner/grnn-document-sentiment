import os
import pickle
import string
from typing import Tuple, Dict, Union

import numpy as np
import pandas as pd
import gensim as gs

from DocSenTypes import *


class Word2Vector:
    def __init__(self, data_paths: Union[str, List[str]], w2v_path: str, name: str, overwrite: bool=False,
                 dim: int = 200):
        self._data_paths = data_paths
        self._w2v_path = w2v_path
        self._name = name
        self._overwrite = overwrite
        self._dim = dim

        self.unknown_word_key = '__UNK__'
        self.padding_word_key = '__PAD__'

    def get_embedding(self, docs: List[TDocumentStr]) \
            -> Tuple[TEmbedding, Dict[TWord, TVocabIndex]]:

        if self._overwrite or (not os.path.isfile(self._w2v_model_path())):
            print("No persisted word2vec model found. Creating a new embedding...")
            self._make_embedding(docs)
        print("Loading word2vec model...")
        return self._load_embedding()

    def _make_embedding(self, docs: List[TDocumentStr]):
        sentences = [sentence for doc in docs for sentence in doc]
        model = gs.models.Word2Vec(sentences, size=self._dim, window=5)
        model.save(self._w2v_model_path())

    def _load_embedding(self) -> Tuple[TEmbedding, Dict[TWord, TVocabIndex]]:
        """
        Load word embedding from the given word2vec model and extend it with vectors for unknown words and padding.
        :param w2v_model_name: Name of the word2vec model to use
        :return: Word embedding matrix and word2index mapping
        """
        if not os.path.isfile(self._w2v_model_path()):
            raise FileNotFoundError(f"Can't find a Word2Vec model with name '{self._name}' on path '{self._w2v_path}'")

        model = gs.models.KeyedVectors.load(self._w2v_model_path())
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

        word2index[self.unknown_word_key] = len(embedding) - 2  # map unknown words to vector we just appended
        word2index[self.padding_word_key] = len(embedding) - 1

        return embedding, word2index

    def _w2v_model_path(self):
        return os.path.join(self._w2v_path, f'{self._name}_w2v_model')

    def _w2v_corpus_path(self):
        return os.path.join(self._w2v_path, f'{self._name}_w2v_corpus_train')
