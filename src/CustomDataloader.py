from torch.utils.data import sampler, Dataset
from typing import List


class CustomDataloader:
    """
    On iteration it yields batches with desired batch size, generated using the given sampler.
    These batches can then again be iterated to receive tuples of document and label.
    """
    def __init__(self, batch_size: int, sampler: sampler.Sampler, dataset: Dataset):
        self._batch_size = batch_size
        self._sampler = sampler
        self._num_batches = len(self._sampler) // batch_size
        self._dataset = dataset

    def __iter__(self):
        i = 0
        batch_idx = []
        for doc_idx in self._sampler:
            batch_idx.append(doc_idx)
            i += 1
            if i % self._batch_size == 0:
                yield self._batch_iterator(batch_idx)
                batch_idx = []

    def __len__(self):
        return len(self._sampler) // self._batch_size

    def _batch_iterator(self, batch: List[int]):
        for i in batch:
            (doc, label) = self._dataset[i]
            yield (doc, label)
