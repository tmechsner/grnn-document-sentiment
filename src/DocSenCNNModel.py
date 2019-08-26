import torch
from torch.utils.data import sampler, DataLoader

import numpy as np

from src.ImdbDataset import ImdbDataset

from src.DocumentPadCollate import DocumentPadCollate


class DocSenModel(torch.nn.Module):
    def __init__(self, embedding_matrix: np.array, freeze_embedding: bool = True):
        super(DocSenModel, self).__init__()

        self._word_embedding_dim = len(embedding_matrix[0])
        self._vocab_size = len(embedding_matrix)

        self._word_embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(embedding_matrix), freeze=freeze_embedding)
        self._conv1 = torch.nn.Conv1d(200, 50, 1, stride=1)
        self._conv2 = torch.nn.Conv1d(200, 50, 2, stride=1)
        self._conv3 = torch.nn.Conv1d(200, 50, 3, stride=1)
        self._conv = [self._conv1, self._conv2, self._conv3]

        self._tanh = torch.nn.Tanh()

    def forward(self, doc, pad_vector):
        """
        Process single document
        :param doc:
        :param pad_vector:
        :return:
        """

        num_sentences = len(doc)
        for i in range(0, num_sentences):
            sentence = self._word_embedding(torch.tensor(doc[i]))
            num_words = len(sentence)

            sentence = sentence.unsqueeze(2)  # Add third dimension for number of sentences (here: always equal to one)
            sentence = sentence.permute(2, 1, 0)

            conv = []
            for kernel_size in range(1, 4):
                if num_words >= kernel_size:
                    pool_layer = torch.nn.AvgPool1d(num_words - kernel_size + 1)
                    pool_layer.double()

                    X = self._conv[kernel_size - 1](sentence)
                    X = pool_layer(X)
                    X = self._tanh(X)
                    conv.append(X)
                else:
                    break
            X = torch.cat((conv[0], conv[1], conv[2])).mean(0)



        return doc


def get_batch(batch_size, sampler: sampler.Sampler):
    result = []
    i = 0
    for doc in sampler:
        result.append(doc)
        i += 1
        if i == batch_size:
            break
    return result


def train(batch_size, dataset, learning_rate, model, num_epochs, random_seed, shuffle_dataset, validation_split):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = sampler.SubsetRandomSampler(train_indices)
    valid_sampler = sampler.SubsetRandomSampler(val_indices)

    # DataLoader srews up the data if they are of varying length.
    # Like our data: documents have varying number of sentences and sentences have varying number of words.
    # Thus we need to use a custom collate function that pads the documents' sentence lists and every sentence's word
    # list before the document tensors get stacked.

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    learning_curve = []
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch}')

        num_batches = len(train_indices) // batch_size
        for i in range(num_batches):
            batch = get_batch(batch_size, train_sampler)
            docs = []
            labels = []
            for j in batch:
                (doc, label) = dataset[j]
                docs.append(doc)
                labels.append(label)
                # apply the model with the current parameters
                label_predicted = model(doc, dataset.embedding[dataset.word2index[dataset.padding_word_key]])

            # compute the loss and store it; note that the loss is an object
            # which we will also need to compute the gradient
            # loss_object = loss_function(labels[i], label_predicted)
            # learning_curve.append(loss_object.item())
            #
            # # print the loss every 50 steps so that we see the progress
            # # while learning happens
            # if len(learning_curve) % 50 == 0:
            #     print('loss after {} steps: {}'.format(len(learning_curve), learning_curve[-1]))
            #
            # # A special feature of PyTorch is that we need to zero the gradients
            # # in the optimizer to ensure that past computations do
            # # not influence the present ones
            # optimizer.zero_grad()
            #
            # # compute the gradient of the loss
            # loss_object.backward()
            #
            # # compute a step of the optimizer based on the gradient
            # optimizer.step()


def main():
    num_epochs = 10
    w2v_sample_frac = 0.9
    data_path = 'data/Dev/imdb-dev.txt.ss'
    data_name = 'imdb-dev'
    freeze_embedding = True
    batch_size = 16
    validation_split = 0.2
    shuffle_dataset = False
    random_seed = 42
    learning_rate = 0.03

    dataset = ImdbDataset(data_path, data_name, w2v_sample_frac=w2v_sample_frac)

    model = DocSenModel(dataset.embedding, freeze_embedding=freeze_embedding)
    model.double()

    train(batch_size, dataset, learning_rate, model, num_epochs, random_seed, shuffle_dataset, validation_split)


if __name__ == '__main__':
    main()
