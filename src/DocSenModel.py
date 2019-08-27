from enum import Enum
import torch
import numpy as np


class DocSenModel(torch.nn.Module):
    class SentenceModel(Enum):
        LSTM = 1
        CONV = 2

    class GnnOutput(Enum):
        LAST = 1
        AVG = 2

    class GnnType(Enum):
        FORWARD = 1
        FORWARD_BACKWARD = 2

    def __init__(self, sentence_model: SentenceModel, gnn_output: GnnOutput, gnn_type: GnnType,
                 embedding_matrix: np.array, freeze_embedding: bool = False):
        super(DocSenModel, self).__init__()

        self._sentence_model = sentence_model
        self._gnn_output = gnn_output
        self._gnn_type = gnn_type

        self._word_embedding_dim = len(embedding_matrix[0])
        self._vocab_size = len(embedding_matrix)

        self._word_embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(embedding_matrix), freeze=freeze_embedding)

        if self._sentence_model == self.SentenceModel.CONV:
            self._conv1 = torch.nn.Conv1d(200, 50, 1, stride=1)
            self._conv2 = torch.nn.Conv1d(200, 50, 2, stride=1)
            self._conv3 = torch.nn.Conv1d(200, 50, 3, stride=1)
            self._conv = [self._conv1, self._conv2, self._conv3]
        else:
            self._lstm = torch.nn.LSTM(200, 50, num_layers=1)

        self._tanh = torch.nn.Tanh()

        self.double()

    def forward(self, doc):
        """
        Process a single document
        :param doc:
        :return:
        """

        num_sentences = len(doc)
        for i in range(0, num_sentences):
            # Turn vocabulary ids into embedding vectors
            sentence = self._word_embedding(torch.tensor(doc[i], dtype=torch.long))
            num_words = len(sentence)

            if num_words == 0:
                continue

            # Add third dimension for number of sentences (here: always equal to one)
            sentence = sentence.unsqueeze(2)

            # Model the sentences either with convolutional filters or with an LSTM
            if self._sentence_model == self.SentenceModel.CONV:
                sentence_rep = self._sentence_convolution(num_words, sentence)
            else:
                sentence_rep = self._sentence_lstm(sentence)

            # Todo: GRNN

            X = sentence_rep

        return X

    def _sentence_convolution(self, num_words, sentence):
        # Rearrange shape for Conv1D layers
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
        return sentence_rep

    def _sentence_lstm(self, sentence):
        sentence = sentence.permute(0, 2, 1)
        initial_hidden_state = (torch.randn(1, 1, 50, dtype=torch.double),
                                torch.randn(1, 1, 50, dtype=torch.double))
        out, _ = self._lstm(sentence, initial_hidden_state)

        # LSTM output contains the whole state history for this sentence.
        # We only need the last output.
        sentence_rep = out[-1]
        return sentence_rep
