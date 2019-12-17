# encoding:utf8

import numpy as np
import pandas as pd


def build_dataset(doc_embedd, news_dataset):
    """
    Build train dataset
    :param doc_embedd  doc embedding vector
    :para news_dataset news dataset
    :return X, y
    """

    doc_indexes = []
    doc_embeddings = []

    with open(doc_embedd) as f:
        first_line = True
        for l in f.readlines():
            # skip first line
            if first_line:
                first_line = False
                continue

            embedding = l.strip('\n').split(' ')
            doc_index, doc_embedding = embedding[0].split('_')[1], embedding[1:]
            doc_indexes.append(doc_index)
            doc_embeddings.append(doc_embedding)

    X = np.array(doc_embeddings)

    news_df = pd.read_csv(news_dataset)
    y = news_df.apply(lambda row: 1 if row['source'] == '新华社' else 0, axis=1)
    y = np.array(y[:len(doc_embeddings)])

    return X, y
