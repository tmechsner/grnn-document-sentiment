import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import sampler

from DocSenModel import DocSenModel

from ImdbDataloader import ImdbDataloader
from ImdbDataset import ImdbDataset

import matplotlib.pyplot as plt

from YelpDataset import YelpDataset


def split_data(dataset, random_seed, shuffle_dataset, validation_split):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    return indices[split:], indices[:split]


def train(batch_size, dataset, learning_rate, model, num_epochs, random_seed, shuffle_dataset, validation_split,
          model_path, continue_training=True):

    loss_function = torch.nn.CrossEntropyLoss()

    # Todo: Which optimizer to use? In Paper: Simple SGD, no Momentum
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    checkpoint_path = model_path + '_checkpoint.tar'
    if continue_training and os.path.isfile(checkpoint_path):
        print("Loading checkpoint to continue training...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_0 = checkpoint['epoch'] + 1
        train_loss = checkpoint['train_loss']
        valid_loss = checkpoint['valid_loss']
        train_indices = checkpoint['train_indices']
        val_indices = checkpoint['val_indices']
        print(f"Continue training in epoch {epoch_0+1}")
    else:
        print("Not loading a training checkpoint.")
        train_loss = []
        valid_loss = []
        epoch_0 = 0
        train_indices, val_indices = split_data(dataset, random_seed, shuffle_dataset, validation_split)

    train_sampler = sampler.SubsetRandomSampler(train_indices)
    valid_sampler = sampler.SubsetRandomSampler(val_indices)

    dataloader_train = ImdbDataloader(batch_size, train_sampler, dataset)
    dataloader_valid = ImdbDataloader(batch_size, valid_sampler, dataset)

    for epoch in range(epoch_0, num_epochs):
        print(f'\nEpoch {epoch+1} of {num_epochs}')

        for batch_num, batch in enumerate(dataloader_train):
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
            train_loss.append(loss_object.item())

            # Reset the gradients in the optimizer.
            # Otherwise past computations would influence new computations.
            optimizer.zero_grad()
            loss_object.backward()
            optimizer.step()

            # Test on a single batch from the validation set
            # Set model to evaluation mode
            model.eval()
            batch_valid_indices = random.choices(val_indices, k=batch_size)
            for (doc, label) in dataloader_valid._batch_iterator(batch_valid_indices):
                prediction = model(doc)
                prediction = prediction.unsqueeze(0)
                predictions = prediction if predictions is None else torch.cat((predictions, prediction))
                label = torch.Tensor([label])
                label = label.long()
                labels = label if labels is None else torch.cat((labels, label))

            # Compute the loss
            loss_object = loss_function(predictions, labels)
            valid_loss.append(loss_object.item())

            # Set model back to training mode
            model.train()

            if batch_num % 10 == 0:
                print(f"  Batch {batch_num+1} of {len(dataloader_train)}. Training-Loss: {train_loss[-1]} \t"
                      f" Validation-Loss: {valid_loss[-1]}")

        print("Saving training progress checkpoint...")
        if os.path.isfile(checkpoint_path):
            os.rename(checkpoint_path, checkpoint_path + '_old')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'train_indices': train_indices,
            'val_indices': val_indices
        }, checkpoint_path + '_tmp')
        os.rename(checkpoint_path + '_tmp', checkpoint_path)
        if os.path.isfile(checkpoint_path + '_old'):
            os.remove(checkpoint_path + '_old')


def validate(dataset, model, model_path):

    checkpoint_path = model_path + '_checkpoint.tar'
    if not os.path.isfile(checkpoint_path):
        print("Couldn't find the model checkpoint.")
        return

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    valid_loss = checkpoint['valid_loss']
    train_indices = checkpoint['train_indices']
    val_indices = checkpoint['val_indices']

    valid_sampler = sampler.SubsetRandomSampler(val_indices)

    print(f"Calculating accuracy of the model after {epoch} epochs of training...")

    matches = 0
    for k, i in enumerate(val_indices):
        if k % 10 == 0:
            print(f"Data sample {k+1} of {len(val_indices)}")
        (doc, label) = dataset[i]
        prediction = torch.argmax(model(doc))
        label = torch.Tensor([label])
        label = label.long()
        if label == prediction:
            matches += 1
    accuracy = float(matches) / float(len(val_indices))
    print(f"Accuracy: {accuracy}")


def plot_loss_up_to_checkpoint(model_path, smoothing_window=300):
    checkpoint_path = model_path + '_checkpoint.tar'
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        train_loss = checkpoint['train_loss']
        valid_loss = checkpoint['valid_loss']
        fig = plt.figure()
        plt.plot(range(len(train_loss)), pd.Series(train_loss).rolling(window=smoothing_window).mean().values,
                 label='Training Loss')
        plt.plot(range(len(valid_loss)), pd.Series(valid_loss).rolling(window=smoothing_window).mean().values,
                 label='Validation Loss')
        plt.legend()
        plt.xlabel('Batch')
        plt.ylabel('Cross-Entropy Loss')
        plt.show()
        plt.close(fig)


def main():
    # Set model name for persistence here
    model_path = '../models/gnn-conv-avg-forward-backward-yelp-word-linear-bs-50-lr-0.03'

    # Specify what you want to do:
    # Plot the loss up to the most recent checkpoint?
    # Train the model?
    # Or validate accuracy? (Both options = false)
    plot_loss = False
    train_model = True

    if plot_loss:
        plot_loss_up_to_checkpoint(model_path, smoothing_window=20)
        quit()
    else:
        num_epochs = 70
        w2v_sample_frac = 0.9
        # data_path = '../data/Dev/imdb-dev.txt.ss'
        data_path = '../data/Yelp/2013_witte/yelp_academic_dataset_review.json'
        data_name = 'yelp'
        freeze_embedding = True
        batch_size = 50
        validation_split = 0.2
        shuffle_dataset = False

        random_seed = 3
        learning_rate = 0.03

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # dataset = ImdbDataset(data_path, data_name, w2v_sample_frac=w2v_sample_frac, use_reduced_dataset=0)
        dataset = YelpDataset(data_path, data_name, w2v_sample_frac=w2v_sample_frac, use_reduced_dataset=0)

        model = DocSenModel(dataset.num_classes,
                            DocSenModel.SentenceModel.CONV,
                            DocSenModel.GnnOutput.AVG,
                            DocSenModel.GnnType.FORWARD_BACKWARD,
                            dataset.embedding,
                            freeze_embedding)

        if train_model:
            train(batch_size, dataset, learning_rate, model, num_epochs, random_seed, shuffle_dataset, validation_split,
                  model_path)
        else:
            validate(dataset, model, model_path)


if __name__ == '__main__':
    main()
