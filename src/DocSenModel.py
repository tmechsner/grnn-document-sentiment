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

    def __init__(self, sentence_model: SentenceModel, gnn_output: GnnOutput, gnn_type: GnnType, embedding_matrix: np.array, freeze_embedding: bool = False):
        super(DocSenModel, self).__init__()

        self._sentence_model = sentence_model
        self._gnn_output = gnn_output
        self._gnn_type = gnn_type

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
        Process a single document
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


class DocSenModelBuilder:
    def __init__(self, embedding_matrix: np.array):
        self.embedding_matrix = embedding_matrix

    def with_lstm_sentences(self):
        """
        Model sentences using an LSTM on the word vectors.
        """
        return DocSenModelBuilder._DocSenModelBuilderA(self, DocSenModel.SentenceModel.LSTM)

    def with_conv_sentences(self):
        """
        Model sentences using three parallel convolutional filters with different kernel size.
        """
        return DocSenModelBuilder._DocSenModelBuilderA(self, DocSenModel.SentenceModel.CONV)

    class _DocSenModelBuilderA:
        def __init__(self, _0, conv_or_lstm: DocSenModel.SentenceModel):
            self._0 = _0
            self._conv_or_lstm = conv_or_lstm

        def with_gnn_last(self):
            """
            Use only the last output of the sentence GNN chain.
            """
            return DocSenModelBuilder._DocSenModelBuilderB(self._0, self, DocSenModel.GnnOutput.LAST)

        def with_gnn_avg(self):
            """
            Use an average of all outputs of the sentence GNN chain.
            """
            return DocSenModelBuilder._DocSenModelBuilderB(self._0, self, DocSenModel.GnnOutput.AVG)

    class _DocSenModelBuilderB:
        def __init__(self, _0, a, gnn_output: DocSenModel.GnnOutput):
            self._0 = _0
            self._a = a
            self._gnn_output = gnn_output

        def with_forward_gnn(self):
            """
            Process sentences in a forward GNN only.
            """
            return DocSenModelBuilder._DocSenModelBuilderC(self._0, self._a, self, DocSenModel.GnnType.FORWARD)

        def with_forward_backward_gnn(self):
            """
            Process sentences in a forward and a backward GNN.
            """
            return DocSenModelBuilder._DocSenModelBuilderC(self._0, self._a, self, DocSenModel.GnnType.FORWARD_BACKWARD)

    class _DocSenModelBuilderC:
        def __init__(self, _0, a, b, gnn_type: DocSenModel.GnnType):
            self._0 = _0
            self._a = a
            self._b = b
            self._gnn_type = gnn_type

        def with_frozen_embedding(self, frozen_embedding: bool = True):
            """
            (Don't) fine tune the word embeddings during training.
            """
            return DocSenModelBuilder._DocSenModelBuilderD(self._0, self._a, self._b, self, frozen_embedding)

        def build(self):
            return DocSenModel(self._a.conv_or_lstm, self._b.gnn_output, self._gnn_type, self._0.embedding_matrix)

    class _DocSenModelBuilderD:
        def __init__(self, _0, a, b, c, freeze_embedding):
            self._0 = _0
            self._a = a
            self._b = b
            self._c = c
            self._freeze_embedding = freeze_embedding

        def build(self):
            """
            Create a Document Sentiment Model with the specified configuration.
            """
            model = DocSenModel(self._a.conv_or_lstm, self._b.gnn_output, self._c.gnn_type, self._0.embedding_matrix,
                                freeze_embedding=self._freeze_embedding)
            model.double()
            return model
