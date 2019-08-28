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

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataloader = ImdbDataloader(batch_size, train_sampler, dataset)

    learning_curve = []
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch}')

        for batch_num, batch in enumerate(dataloader):


            # Forward pass for each single document in the batch
            predictions = None
            labels = None
            for (doc, label) in batch:
                prediction = model(doc)
                prediction = prediction.unsqueeze(0)
                predictions = prediction if predictions is None else torch.cat((predictions, prediction))
                label = torch.Tensor([label])
                label = label.long()
                labels = label if labels is None else torch.cat((labels, label))

            # Compute the loss
            loss_object = loss_function(predictions, labels)
            learning_curve.append(loss_object.item())

            if batch_num % 10 == 0:
                print(f'  Batch {batch_num+1} of {len(dataloader)}. Loss: {learning_curve[-1]}')

            # Reset the gradients in the optimizer.
            # Otherwise past computations would influence new computations.
            optimizer.zero_grad()
            loss_object.backward()
            optimizer.step()


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

    gnn_conv = DocSenModel(dataset.num_classes,
                           DocSenModel.SentenceModel.CONV,
                           DocSenModel.GnnOutput.AVG,
                           DocSenModel.GnnType.FORWARD,
                           dataset.embedding,
                           freeze_embedding)
    train(batch_size, dataset, learning_rate, gnn_conv, num_epochs, random_seed, shuffle_dataset, validation_split)

    gnn_lstm = DocSenModel(dataset.num_classes,
                           DocSenModel.SentenceModel.LSTM,
                           DocSenModel.GnnOutput.LAST,
                           DocSenModel.GnnType.FORWARD,
                           dataset.embedding,
                           freeze_embedding)
    train(batch_size, dataset, learning_rate, gnn_lstm, num_epochs, random_seed, shuffle_dataset, validation_split)


if __name__ == '__main__':
    main()
