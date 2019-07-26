Kurze Erläuterung:
Ich habe mich bei dem Preprocessing an der Hands-on Session für CNNs orientiert. Für eine embedding-layer braucht es
eine embedding Matrix als Gewichte. Statt der einzelnen Worte werden dann die Indizes in der embedding-matrix als Input
benutzt.
In word2vec_model_imdb ist das komplette word2vec model, trainiert mit gensim (Python package). In der embedding_matrix
ist eine Liste mit den Wortvectoren, geordnet nach dem Index im word2vec-model. In den X-daten ist eine Liste mit den Rezensionen, wobei jede Rezension aus einer Liste von Sätzen besteht. Jeder Satz ist wiederum eine Liste mit Worten,
wobei im Preprocessing die Worte durch ihre Indizes im w2v model ersetzt wurden.

 
