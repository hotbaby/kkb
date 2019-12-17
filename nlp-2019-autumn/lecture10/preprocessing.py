# encoding: utf8

"""
Preprocessing news data.
"""

import jieba
import pandas as pd


def tokenize(s):
    return list(jieba.cut(s))


def gen_corpus(filepath, num=-1):
    news_df = pd.read_csv('news.csv')

    with open(filepath, 'w') as f:
        counter = 0
        for content in news_df['content'].astype(str):
            tokens = tokenize(content)
            f.write(' '.join(tokens).replace('\n', '').strip() + '\n')

            counter += 1
            if num > 0 and counter >= num:
                break

    return None


if __name__ == '__main__':
    gen_corpus('news_corpus.txt', 100)
