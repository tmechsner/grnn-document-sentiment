import torch
import numpy as np


class DocumentPadCollate:

    def __init__(self, padding_word_id, dim=0, dtype=int):
        """
        :param padding_word_id: id of the padding word in the vocabulary
        :param dim: the dimension to be padded
        :param dtype: type of padding zeros
        """
        self.padding_word_id = padding_word_id
        self.dim = dim
        self.dtype = dtype

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            documents - a tensor of all examples in 'batch' after padding
            labels - a LongTensor of all labels in batch
        """

        documents = [np.array(sample[0]) for sample in batch]
        labels = [sample[1] for sample in batch]

        # find document with most sentences
        max_sentences = max([doc.shape[self.dim] for doc in documents])
        max_words = max([np.array(sentence).shape[self.dim] for doc in documents for sentence in doc])

        # pad according to max_sentences
        docs_padded = [self._pad_document(doc, pad_sentences=max_sentences, pad_words=max_words, dim=self.dim) for doc in documents]

        # stack all
        documents_stacked = torch.stack(docs_padded, dim=0)
        labels = torch.from_numpy(np.array(labels))

        return documents_stacked, labels

    def _pad_document(self, document, pad_sentences, pad_words, dim):
        """

        :param document: tensor to pad
        :param pad_sentences: the size to pad the document to
        :param pad_words: the size to pad the sentences to
        :param dim: dimension to pad
        :return: a new tensor padded to 'pad' in dimension 'dim'
        """
        document = [self._pad_sentence(np.array(sentence), pad_words=pad_words, dim=0) for sentence in document if len(sentence) > 0]

        pad_size = pad_sentences - len(document)

        # list of tensors to multidimensional tensor
        document = torch.cat([sentence.unsqueeze(0) for sentence in document])

        return torch.cat([document, torch.zeros(pad_size, pad_words, dtype=self.dtype)], dim=dim)

    def _pad_sentence(self, sentence, pad_words, dim):
        """
        Pad sentence with zero words
        :param sentence: tensor to pad
        :param pad_words: the size to pad to
        :param dim: dimension to pad
        :return: a new tensor padded to 'pad' in dimension 'dim'
        """
        pad_size = list(sentence.shape)
        pad_size[dim] = pad_words - sentence.shape[dim]
        sentence = torch.from_numpy(sentence)

        return torch.cat([sentence, self.padding_word_id * torch.ones(*pad_size, dtype=self.dtype)], dim=dim)

    def __call__(self, batch):
        return self.pad_collate(batch)
