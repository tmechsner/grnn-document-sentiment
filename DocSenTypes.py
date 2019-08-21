from numpy import array
from typing import List

TWord = str
TSentenceStr = List[TWord]
TDocumentStr = List[TSentenceStr]

TVocabIndex = int
TSentenceInd = List[TVocabIndex]
TDocumentInd = List[TSentenceInd]

TRating = int
TEmbedding = array
