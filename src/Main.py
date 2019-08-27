import numpy as np
import torch
from torch.utils.data import sampler

from src.DocSenModel import DocSenModel

from src.ImdbDataloader import ImdbDataloader
from src.ImdbDataset import ImdbDataset


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
            if batch_num % 10 == 0:
                print(f'  Batch {batch_num+1} of {len(dataloader)}')

            for (doc, label) in batch:
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
    data_path = '../data/Dev/imdb-dev.txt.ss'
    data_name = 'imdb-dev'
    freeze_embedding = True
    batch_size = 16
    validation_split = 0.2
    shuffle_dataset = False
    random_seed = 42
    learning_rate = 0.03

    dataset = ImdbDataset(data_path, data_name, w2v_sample_frac=w2v_sample_frac)

    gnn_conv = DocSenModel(DocSenModel.SentenceModel.CONV,
                           DocSenModel.GnnOutput.LAST,
                           DocSenModel.GnnType.FORWARD,
                           dataset.embedding,
                           freeze_embedding)
    train(batch_size, dataset, learning_rate, gnn_conv, num_epochs, random_seed, shuffle_dataset, validation_split)

    gnn_lstm = DocSenModel(DocSenModel.SentenceModel.LSTM,
                           DocSenModel.GnnOutput.LAST,
                           DocSenModel.GnnType.FORWARD,
                           dataset.embedding,
                           freeze_embedding)
    train(batch_size, dataset, learning_rate, gnn_lstm, num_epochs, random_seed, shuffle_dataset, validation_split)


if __name__ == '__main__':
    main()
