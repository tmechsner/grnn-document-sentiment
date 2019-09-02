# From within /src call:
# floyd run --env pytorch-1.0 --data deratomkeks/datasets/yelp-2013-academic/2:yelp --data deratomkeks/projects/grnn-document-sentiment/15:/checkpoint "python3 Main.py"
# where 15 here is the number of the run to continue.
# Run the following to start a new training:
# floyd run --env pytorch-1.0 --data deratomkeks/datasets/yelp-2013-academic/2:yelp "python3 Main.py"

import os
from shutil import copyfile
import random
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import sampler

from DocSenModel import DocSenModel

from CustomDataloader import CustomDataloader

import matplotlib.pyplot as plt

from ImdbDataset import ImdbDataset
from YelpDataset import YelpDataset


def split_data(dataset, random_seed, shuffle_dataset, validation_split):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    return indices[split:], indices[:split]


def train(batch_size, dataset, learning_rate, lr_decay_factor, model, num_epochs, random_seed, shuffle_dataset, validation_split,
          model_path, continue_training=True, early_stopping=2):

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
        train_acc = checkpoint['train_acc']
        valid_acc = checkpoint['valid_acc']
        train_indices = checkpoint['train_indices']
        val_indices = checkpoint['val_indices']
        try:
            learning_rate = checkpoint['learning_rate']
            lr_decay_factor = checkpoint['lr_decay_factor']
        except Exception:
            pass
        print(f"Continue training in epoch {epoch_0+1}")
    else:
        print("Not loading a training checkpoint.")
        train_loss = []
        valid_loss = []
        train_acc = []
        valid_acc = []
        epoch_0 = 0
        train_indices, val_indices = split_data(dataset, random_seed, shuffle_dataset, validation_split)

    train_sampler = sampler.SubsetRandomSampler(train_indices)
    valid_sampler = sampler.SubsetRandomSampler(val_indices)

    dataloader_train = CustomDataloader(batch_size, train_sampler, dataset)
    dataloader_valid = CustomDataloader(batch_size, valid_sampler, dataset)

    min_loss = 99999
    min_loss_epoch = epoch_0
    for epoch in range(epoch_0, num_epochs):
        print(f'\nEpoch {epoch+1} of {num_epochs}')

        if early_stopping > 0:
            if epoch - min_loss_epoch >= early_stopping:
                print(f"No training improvement over the last {early_stopping} epochs. Aborting.")
                break

        learning_rate = lr_decay_factor * learning_rate

        for batch_num, batch in enumerate(dataloader_train):
            # Forward pass for each single document in the batch
            predictions = None
            labels = None
            matches = 0
            for (doc, label) in batch:
                if len(doc) == 0:
                    continue
                try:
                    prediction = model(doc)
                    prediction = prediction.unsqueeze(0)
                    predictions = prediction if predictions is None else torch.cat((predictions, prediction))
                    label = torch.Tensor([label])
                    label = label.long().to(model._device)
                    labels = label if labels is None else torch.cat((labels, label))

                    if label == torch.argmax(prediction):
                        matches += 1
                except AttributeError as e:
                    print("Some error occurred. Ignoring this document. Error:")
                    print(e)
                    continue

            # Compute the loss
            loss_object = loss_function(predictions, labels)
            loss = loss_object.item()
            train_loss.append(loss)

            accuracy = float(matches) / float(len(predictions))
            train_acc.append(accuracy)

            if loss < min_loss:
                min_loss = loss
                min_loss_epoch = epoch

            # Reset the gradients in the optimizer.
            # Otherwise past computations would influence new computations.
            optimizer.zero_grad()
            loss_object.backward()
            optimizer.step()

            # Test on a single batch from the validation set
            # Set model to evaluation mode
            model.eval()
            batch_valid_indices = random.choices(val_indices, k=batch_size)
            predictions = None
            labels = None
            matches = 0
            for (doc, label) in dataloader_valid._batch_iterator(batch_valid_indices):
                if len(doc) == 0:
                    continue
                try:
                    prediction = model(doc)
                    prediction = prediction.unsqueeze(0)
                    predictions = prediction if predictions is None else torch.cat((predictions, prediction))
                    label = torch.Tensor([label])
                    label = label.long().to(model._device)
                    labels = label if labels is None else torch.cat((labels, label))

                    if label == torch.argmax(prediction):
                        matches += 1
                except AttributeError as e:
                    print("Some error occurred. Ignoring this document. Error:")
                    print(e)
                    continue

            # Compute the loss
            loss_object = loss_function(predictions, labels)
            valid_loss.append(loss_object.item())

            accuracy = float(matches) / float(len(predictions))
            valid_acc.append(accuracy)

            # Set model back to training mode
            model.train()

            if batch_num % 10 == 0:
                print(f"  Batch {batch_num+1:>5} of {len(dataloader_train):>5}  -"
                      f"   Training-Loss: {train_loss[-1]:.4f}   Validation-Loss: {valid_loss[-1]:.4f}"
                      f"   Training-Acc.: {train_acc[-1]:.2f}   Validation-Acc.: {valid_acc[-1]:.2f}")

        print("Saving training progress checkpoint...")
        if os.path.isfile(checkpoint_path):
            os.rename(checkpoint_path, checkpoint_path + '_' + str(epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'train_acc': train_acc,
            'valid_acc': valid_acc,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'learning_rate': learning_rate,
            'lr_decay_factor': lr_decay_factor
        }, checkpoint_path + '_tmp')
        os.rename(checkpoint_path + '_tmp', checkpoint_path)


def evaluate(dataset, model, model_path):

    checkpoint_path = model_path + '_checkpoint.tar'
    if not os.path.isfile(checkpoint_path):
        print("Couldn't find the model checkpoint.")
        return

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    val_indices = checkpoint['val_indices']

    valid_sampler = sampler.SubsetRandomSampler(val_indices)

    print(f"Calculating accuracy of the model after {epoch+1} epochs of training...")

    matches = 0
    diffs = []
    processed_docs = 0
    for k, i in enumerate(val_indices):
        if k % 10 == 0:
            print(f"Data sample {k+1} of {len(val_indices)}")
        (doc, label) = dataset[i]
        try:
            prediction = torch.argmax(model(doc))
        except AttributeError as e:
            print("Something went wrong. Ignoring this document.")
            continue
        label = torch.Tensor([label])
        label = label.long()
        if label == prediction:
            matches += 1
        processed_docs += 1
        diffs.append(int(np.abs(label - prediction)))
    accuracy = float(matches) / float(processed_docs)
    diffs = np.array(diffs)
    mae = diffs.mean()
    aev = diffs.var()
    print(f"Accuracy: {accuracy}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Absolute Error Variance: {aev}")


def plot_loss_up_to_checkpoint(model_path, smoothing_window=300):
    checkpoint_path = model_path + '_checkpoint.tar'
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        train_loss = checkpoint['train_loss']
        valid_loss = checkpoint['valid_loss']
        train_acc = checkpoint['train_acc']
        valid_acc = checkpoint['valid_acc']

        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        # ax1.set_title('Loss')
        ax1.set_ylabel('Cross-Entropy Loss')
        ax1.plot(range(len(train_loss)), pd.Series(train_loss).rolling(window=smoothing_window).mean().values,
                 label='Training')
        ax1.plot(range(len(valid_loss)), pd.Series(valid_loss).rolling(window=smoothing_window).mean().values,
                 label='Validation')

        # ax2.set_title('Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.plot(range(len(train_acc)), pd.Series(train_acc).rolling(window=smoothing_window).mean().values,
                 label='Training')
        ax2.plot(range(len(valid_acc)), pd.Series(valid_acc).rolling(window=smoothing_window).mean().values,
                 label='Validation')

        plt.xlabel('Batch')
        plt.legend()
        plt.show()
        plt.close(f)


def main():
    if not os.path.exists('models'):
        os.makedirs('models')

    num_epochs = 70
    w2v_sample_frac = 0.9
    # data_path = '../data/Dev/imdb-dev.txt.ss'
    data_path = '/yelp/yelp_academic_dataset_review.json'
    data_name = 'yelp'
    freeze_embedding = True
    batch_size = 50
    validation_split = 0.2
    shuffle_dataset = False
    cuda = False

    random_seed = 3
    learning_rate = 0.03
    lr_decay_factor = 0.6

    action = 0

    parser = argparse.ArgumentParser(description="GRNN-Document-Sentiment Model")

    # Actions
    parser.add_argument('-a', '--action', help="Action to be executed (0=train, 1=plot, 2=evaluate)", type=int, default=action)

    # Params
    parser.add_argument('-r', '--random-seed', type=int, default=random_seed)
    parser.add_argument('-l', '--learning-rate', type=float, default=learning_rate)
    parser.add_argument('-u', '--lr-decay-factor', help="After each epoch: lr = lr * u", type=float, default=lr_decay_factor)
    parser.add_argument('-e', '--num-epochs', type=int, default=num_epochs)
    parser.add_argument('-d', '--data-path', type=str, default=data_path)
    parser.add_argument('-n', '--data-name', help="Name of the dataset (for savefile naming)", type=str, default=data_name)
    parser.add_argument('-f', '--retrain-embedding', help="Retrain the word embedding", action='store_true', default=(not freeze_embedding))
    parser.add_argument('-b', '--batch-size', type=int, default=batch_size)
    parser.add_argument('-c', '--cuda', help="Enable cuda support", action='store_true', default=cuda)
    parser.add_argument('-m', '--reduced-dataset', help="For testing purposes. Needs to be between 0 and 1."
                                                        " If > 0, use only two classes and a fraction of"
                                                        " <reduced-dataset> of the data.", action='store_true', default=0)
    # Model architecture
    parser.add_argument('--sentence-model', help="0=convolution, 1=lstm)", type=int, default=0)
    parser.add_argument('--gnn-output', help="0=last, 1=avg)", type=int, default=0)
    parser.add_argument('--gnn-type', help="0=forward, 1=forward-backward (requires gnn-output=1)", type=int, default=0)

    args = parser.parse_args()

    # Set model name for persistence here
    model_name = 'gnn-yelp'

    if args.retrain_embedding:
        print("Retraining the word embedding.")
        model_name += '-retrain-embedding'
        freeze_embedding = False
    else:
        print("Freezing the word embedding.")
        freeze_embedding = True

    if args.sentence_model == 0:
        print("Sentence model: convolution")
        model_name += '-conv'
        sentence_model = DocSenModel.SentenceModel.CONV
    else:
        print("Sentence model: LSTM")
        model_name += '-lstm'
        sentence_model = DocSenModel.SentenceModel.LSTM

    if args.gnn_type == 0:
        print("GNN type: forward")
        model_name += '-forward'
        gnn_type = DocSenModel.GnnType.FORWARD
    else:
        print("GNN type: forward-backward")
        model_name += '-forward-backward'
        gnn_type = DocSenModel.GnnType.FORWARD_BACKWARD

    if args.gnn_output == 0:
        print("GNN output: last")
        model_name += '-last'
        gnn_output = DocSenModel.GnnOutput.LAST
    else:
        print("GNN output: avg")
        model_name += '-avg'
        gnn_output = DocSenModel.GnnOutput.AVG

    model_path = 'models/' + model_name
    checkpoint_path = f"/checkpoint/models/{model_name}_checkpoint.tar"
    if os.path.isfile(checkpoint_path):
        copyfile(checkpoint_path, model_path + '_checkpoint.tar')

    if args.action == 1:
        plot_loss_up_to_checkpoint(model_path, smoothing_window=400)
        quit()
    else:
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

        print(f"Random seed: {args.random_seed}")
        print(f"Reduced dataset: {args.reduced_dataset}")

        # dataset = ImdbDataset(data_path, data_name, w2v_sample_frac=w2v_sample_frac, use_reduced_dataset=0)
        dataset = YelpDataset(args.data_path, args.data_name, w2v_sample_frac=w2v_sample_frac, use_reduced_dataset=args.reduced_dataset,
                              w2v_path="/yelp/", prep_path="/yelp/")

        print(f"Number of classes: {dataset.num_classes}")
        print(f"Batch size: {args.batch_size}")
        print(f"Number of epochs: {args.num_epochs}")
        print(f"Learning rate: {args.learning_rate}")

        if args.cuda:
            print("Using cuda")
        else:
            print("Not using cuda")

        model = DocSenModel(dataset.num_classes, sentence_model, gnn_output, gnn_type, dataset.embedding, freeze_embedding, cuda=args.cuda)

        if args.action == 0:
            train(args.batch_size, dataset, args.learning_rate, lr_decay_factor, model, args.num_epochs,
                  args.random_seed, shuffle_dataset, validation_split, model_path)
        else:
            evaluate(dataset, model, model_path)


if __name__ == '__main__':
    main()
