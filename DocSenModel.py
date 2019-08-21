import torch
import torch.nn as nn

import numpy as np

from Preprocessor import Preprocessor

from typing import Dict

from DocSenTypes import *


class DocSenModel:
    def __init__(self, embedding_matrix: np.array, word2idx: Dict[TWord, TVocabIndex]):
        self._word_embedding_dim = len(embedding_matrix[0])
        self._vocab_size = len(embedding_matrix)

        embed = nn.Embedding(self._vocab_size, self._word_embedding_dim, _weight=torch.from_numpy(embedding_matrix))
        # embed(torch.from_numpy(np.array([id]))) => embedding vector of word with given vocab id
        pass


if __name__ == '__main__':
    p = Preprocessor()
    embedding_matrix, word2idx = p.get_embedding('imdb+yelp')
    model = DocSenModel(embedding_matrix, word2idx)
