import os
import pandas as pd
from gensim.models.word2vec import Word2Vec
from utils import extract_method_structured_features, tokenize


def build_vocab(conf):

    projects = ['swt', 'tomcat', 'aspectj', 'jdt', 'birt']
    word_corpus = []
    code_corpus = []
    # word_vocab
    for project in projects:
        # word
        data = pd.read_csv(os.path.join(conf['report_dir'], project + '.csv'))
        # latest revision for code
        commit_id = data.tail(1)['commit_id'].values[0]
        path = os.path.join(conf['output_dir'], project + '/json/' + commit_id + '.json')
        methods = extract_method_structured_features(path)
        code_corpus += (methods['tokens'].tolist() + methods['api'].tolist())

        data['summary'] = data['summary'].apply(tokenize)
        data['description'] = data['description'].apply(tokenize)
        word_corpus += (data['summary'].tolist() + data['description'].tolist() + methods['comment'].tolist())

    w2v = Word2Vec(word_corpus, min_count=conf['min_count'], size=conf['emb_size'])  # unk-> max_token
    word_path = os.path.join(conf['vocab_dir'], 'w2v_512')
    w2v.save(word_path)

    c2v = Word2Vec(code_corpus, min_count=conf['min_count'], size=conf['emb_size'])
    code_path = os.path.join(conf['vocab_dir'], 'c2v_512')
    c2v.save(code_path)
