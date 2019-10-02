# grnn-document-sentiment
Document Modeling with a Gated Recurrent Neural Network for Sentiment Classification

Original Paper:
https://pdfs.semanticscholar.org/82ad/2ca5fdca7e6721c0fc1d97d029b25a11328b.pdf

## Setup
Setup the following file structure within the data directory:

``` 
grnn-document-sentiment/
 |- models/
 |- src/
 |- data/
     |- Preprocessed/ (will contain preprocessed data)  
     |- Word2Vec/ (Will contain W2V model data)
     |- Yelp/
         |- 2013_witte/
              |- yelp_academic_dataset_review.json
```

## Usage
Run with `python Main.py`.
If available it will continue training with the last checkpoint generated with the specified architecture configuration.
Checkpoints are generated after each epoch.
On the first run it will preprocess the data and create the Word2Vec model and save them in the data directory.

**Parameters:**
* `-a` / `--action`: Action to be executed (0=train, 1=plot loss, 2=evaluate). default=0
* `-s` / `--plot-smoothing`: Window size of moving average smoothing. default=50
* `--floyd`: If given, paths are set to work on floyd (for computing in the cloud). default=False
* `-r` / `--random-seed`: default=3
* `-l` / `--learning-rate`: default=0.03
* `-d` / `--lr-decay-factor`: After each epoch: lr = lr * d. default=0.8
* `-g` / `--l2-reg-factor`: L2 regularization. default=1e-5
* `-e` / `--num-epochs`: Maximum number of epochs to train. default=70
* `-f` / `--retrain-embedding`: Retrain the word embedding. default=False
* `-b` / `--batch-size`: default=50
* `-c` / `--cuda`: Enable cuda support. default=False
* `-m` / `--reduced-dataset`: For testing purposes. Needs to be between 0 and 1. If > 0, use only two classes and a fraction of <reduced-dataset> of the data. default=False
* `--sentence-model`: 0=convolution, 1=lstm. default=0
* `--gnn-output`: 0=last, 1=avg. default=0
* `--gnn-type`: 0=forward, 1=forward-backward (requires gnn-output=1). default=0
