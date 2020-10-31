import os
import gc
import logging
import pandas as pd
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from joblib import Parallel, delayed
from utils import *


def load_vocab(vocab_dir):
    w2v = Word2Vec.load(os.path.join(vocab_dir, 'w2v_512'))
    c2v = Word2Vec.load(os.path.join(vocab_dir, 'c2v_512'))
    return w2v, c2v


def word2id(w2v, series):
    vocab = w2v.wv.vocab
    max_token = w2v.wv.vectors.shape[0]

    def token2index(token_list):
        index_list = []
        for token in token_list:
            token_index = vocab[token].index if token in vocab else max_token
            index_list.append(token_index)
        return index_list

    series = series.apply(token2index)  # word tokens to indexes
    return series


def code2id(c2v, series):
    vocab = c2v.wv.vocab
    max_token = c2v.wv.vectors.shape[0]  # 字典的单词总量

    def token2index(token_list):
        index_list = []
        for token in token_list:
            index_list.append(vocab[token].index if token in vocab else max_token)
        return index_list

    series = series.apply(token2index)  # code tokens to indexes
    return series


def prepare_data(conf):
    logger = logging.getLogger(conf['log'])

    # load related files
    repo = pd.read_csv(conf['report'])
    repo['report'] = repo.apply(lambda row: combine_smy_des(row['summary'], row['description']), axis=1)
    repo['commit_time'] = repo['commit_time'].apply(format_time)
    w2v, c2v = load_vocab(conf['vocab_dir'])
    c2c, m2m, mcm, mcg = load_sim_relations(conf['sim_dir'], conf['target'])

    # function to prepare data from each method
    def process(row):
        sub_features = []
        commit_id = row['commit_id'].values[0]
        commit_time = row['commit_time'].values[0]
        faulty_methods = eval(row['method'].values[0])
        revision_methods = os.path.join(conf['json_dir'], commit_id + '.json')

        if not os.path.exists(revision_methods):
            return []

        methods = extract_method_structured_features(revision_methods)

        # check whether all fixed methods exist in extracted code revision
        label = 1
        extracted_methods = methods.index.tolist()
        for m in faulty_methods:
            if m not in extracted_methods:
                label = 0
                break
        if not label:
            return []

        # token to index
        methods['token_id'] = code2id(c2v, methods['tokens'])  # code tokens tokens to index
        methods['api_id'] = code2id(c2v, methods['api'])
        methods['comment_id'] = word2id(w2v, methods['comment'])  # comment tokens to index

        # get previous reports
        prev_reports = repo[repo['commit_time'] < commit_time]

        # compute revised collaborative filtering score
        rcfs_dict = collaborative_filtering_score(row, prev_reports, 100, True, c2c, True, m2m)

        # positive examples
        for faulty_method in faulty_methods:
            row = methods[methods.index == faulty_method]
            token_id = row['token_id'].values[0]
            if len(token_id) == 0:
                continue
            api_id = row['api_id'].values[0]
            comment_id = row['comment_id'].values[0]
            length = row['num_statements'].values[0]
            bfr, bff = calculate_bfr_and_bff(faulty_method, commit_time, prev_reports)
            cfs = rcfs_dict[faulty_method] if faulty_method in rcfs_dict else 0.0
            if length <= 5:
                relevant_methods = get_relevant_methods(faulty_method, m2m, mcm, mcg)
            else:
                relevant_methods = []

            line = [commit_id, faulty_method, token_id, api_id, comment_id, bfr, bff, cfs,
                    length, relevant_methods, 1]
            sub_features.append(line)

        # negative
        negative_methods = list(set(methods.index.tolist()) - set(faulty_methods))
        for negative_method in negative_methods:
            bfr, bff = calculate_bfr_and_bff(negative_method, commit_time, prev_reports)
            cfs = rcfs_dict[negative_method] if negative_method in rcfs_dict else 0.0
            negative_method_row = methods[methods.index == negative_method]
            token_id = negative_method_row['token_id'].values[0]
            if len(token_id) == 0:
                continue
            api_id = negative_method_row['api_id'].values[0]
            comment_id = negative_method_row['comment_id'].values[0]
            length = negative_method_row['num_statements'].values[0]
            if length <= conf['augment_threshold']:
                relevant_methods = get_relevant_methods(negative_method, m2m, mcm, mcg)
                # assert negative_method not in relevant_methods
            else:
                relevant_methods = []
            line = [commit_id, negative_method, token_id, api_id, comment_id, bfr, bff, cfs,
                    length, relevant_methods, 0]
            sub_features.append(line)

        return sub_features

    # split the data into k fold
    num_of_subset = conf['k_fold']
    step = len(repo) // num_of_subset
    start = 0

    for i in range(num_of_subset):
        file_name = 'fold' + '_' + str(i + 1) + '.pkl'
        data_save_path = os.path.join(conf['data_dir'], file_name)
        if os.path.exists(data_save_path):
            print(data_save_path, 'exists!')
            continue

        end = start + step
        sub_repo = repo[start:end]
        start = end
        repo_grouped = sub_repo.groupby(sub_repo.index)
        subsets = Parallel(n_jobs=6)(delayed(process)(row) for index, row in tqdm(repo_grouped))
        features = []
        for sub in subsets:
            features += sub

        features = pd.DataFrame(features,
                                columns=['commit_id', 'method_name', 'token', 'api',
                                         'comment', 'bfr', 'bff', 'cfs', 'statements',
                                         'relevant_methods', 'label'])

        logger.info('subset:{} , total length {}'.format(i + 1, len(features)))
        features.to_pickle(data_save_path)
        del features
        gc.collect()
