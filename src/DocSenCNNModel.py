import torch
from torch.utils.data import sampler

import numpy as np

from src.ImdbDataloader import ImdbDataloader
from src.ImdbDataset import ImdbDataset


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

    def forward(self, doc):
        """
        Process single document
        :param doc:
        :return:
        """

        num_sentences = len(doc)
        for i in range(0, num_sentences):
            # Turn vocabulary ids into embedding vectors
            sentence = self._word_embedding(torch.tensor(doc[i]))
            num_words = len(sentence)

            # Add third dimension for number of sentences (here: always equal to one)
            sentence = sentence.unsqueeze(2)

            # And rearrange shape for Conv1D layers
            sentence = sentence.permute(2, 1, 0)

            # We can't apply a convolution filter to an input that is smaller than the kernel size.
            # Hence, we apply one filter after the other with increasing kernel size until it exceeds input size.
            conv_result = None
            for kernel_size in range(1, 4):
                if num_words >= kernel_size:
                    # Since the size of the sentences varies, we have to rebuild the avg pooling layer every iteration
                    avg_pool_layer = torch.nn.AvgPool1d(num_words - kernel_size + 1)
                    avg_pool_layer.double()

                    X = self._conv[kernel_size - 1](sentence)
                    X = avg_pool_layer(X)
                    X = self._tanh(X)

                    # Concatenate results
                    conv_result = X if conv_result is None else torch.cat((conv_result, X))
                else:
                    break

            # In the end merge the output of all applied pooling layers by averaging them
            sentence_rep = conv_result.mean(0)



        return X


def split_data(dataset, random_seed, shuffle_dataset, validation_split):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    return indices[split:], indices[:split]


def train(batch_size, dataset, learning_rate, model, num_epochs, random_seed, shuffle_dataset, validation_split):

    train_indices, val_indices = split_data(dataset, random_seed, shuffle_dataset, validation_split)

    train_sampler = sampler.SubsetRandomSampler(train_indices)
    valid_sampler = sampler.SubsetRandomSampler(val_indices)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataloader = ImdbDataloader(batch_size, train_sampler, dataset)

    learning_curve = []
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch}')

        for batch_num, batch in enumerate(dataloader):
            for (doc, label) in batch:
                # apply the model with the current parameters
                label_predicted = model(doc)

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
