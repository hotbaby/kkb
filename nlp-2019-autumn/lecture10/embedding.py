# encoding: utf8

import time
import gensim


def read_corpus(fname, sampling=-1):
    with open(fname) as f:
        for i, line in enumerate(f):
            if sampling > 0 and i >= sampling:
                return

            tokens = gensim.utils.simple_preprocess(line)
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


def doc_embedding(corpus_fname, embedd_model_fname, vector_size=100, sampling=-1):
    corpus_docs = list(read_corpus(corpus_fname, sampling))
    model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size)
    model.build_vocab(corpus_docs)
    model.train(corpus_docs, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(embedd_model_fname+'.model')
    model.docvecs.save_word2vec_format(embedd_model_fname+'.dv')

    return model


if __name__ == '__main__':
    doc_embedding('news_corpus.txt', 'doc2vec_{}'.format(int(time.time())))
