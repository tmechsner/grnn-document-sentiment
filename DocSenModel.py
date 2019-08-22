import torch
import torch.nn as nn
from torch.utils.data import sampler, DataLoader

import numpy as np

from ImdbDataset import ImdbDataset

from typing import Dict

from DocSenTypes import *


class DocSenModel:
    def __init__(self, embedding_matrix: np.array, word2idx: Dict[TWord, TVocabIndex]):
        self._word_embedding_dim = len(embedding_matrix[0])
        self._vocab_size = len(embedding_matrix)

        # Call: embed(torch.from_numpy(np.array([id1, id2, ..., idn]))) => n-dim embedding vector of words with given vocab ids
        embed = nn.Embedding.from_pretrained(torch.from_numpy(embedding_matrix))

        pass


if __name__ == '__main__':
    dataset = ImdbDataset('data/Dev/imdb-dev.txt.ss', 'imdb-dev', w2v_sample_frac=0.9)

    model = DocSenModel(dataset.embedding, dataset.word2index)

    batch_size = 16
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = sampler.SubsetRandomSampler(train_indices)
    valid_sampler = sampler.SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    # Usage Example:
    num_epochs = 10
    for epoch in range(num_epochs):
        # Train:
        for batch_index, (documents, labels) in enumerate(train_loader):
            pass
