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
                 dim: int = 200, sample_frac: float = 0.3):
        self._data_paths = data_paths
        self._w2v_path = w2v_path
        self._name = name
        self._overwrite = overwrite
        self._sample_frac = sample_frac
        self._dim = dim

        self.unknown_word_key = '__UNK__'
        self.padding_word_key = '__PAD__'

    def get_embedding(self) \
            -> Tuple[TEmbedding, Dict[TWord, TVocabIndex]]:

        if self._overwrite or (not os.path.isfile(self._w2v_model_path())):
            print("No persisted word2vec model found. Creating a new embedding...")
            self._make_embedding()
        print("Loading word2vec model...")
        return self._load_embedding()

    def _make_embedding(self):
        data = pd.DataFrame()
        if type(self._data_paths) == List[str]:
            for path in self._data_paths:
                data_in = pd.read_csv(path, sep="    ", header=None, engine='python', names=["full_review"])
                sample = data_in.sample(frac=self._sample_frac, replace=False)
                data = pd.concat([data, sample], axis=0)
                del data_in
        else:
            data_in = pd.read_csv(self._data_paths, sep="    ", header=None, engine='python', names=["full_review"])
            data = data_in.sample(frac=self._sample_frac, replace=False)

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
        with open(self._w2v_corpus_path(), "wb") as outfile:
            pickle.dump(sentences_list, outfile)
        model = gs.models.Word2Vec(sentences_list, size=self._dim, window=5)
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
        return os.path.join(self._w2v_path, f'{self._name}_{self._sample_frac}_w2v_model')

    def _w2v_corpus_path(self):
        return os.path.join(self._w2v_path, f'{self._name}_{self._sample_frac}_w2v_corpus_train')
