import gc
import pandas as pd
from train import *
from eval import evaluate


def func(method_names, data, code_pad_idx, word_pad_idx):
    rel_methods = {'token': [], 'api': [], 'comment': []}
    for method in method_names:
        row = data[data['method_name'] == method]
        if len(row):
            rel_methods['token'].append(row['token'].values[0])
            rel_methods['api'].append(row['api'].values[0] + [code_pad_idx])
            rel_methods['comment'].append(row['comment'].values[0] + [word_pad_idx])
    return rel_methods


def within_project_prediction(conf):
    logger = logging.getLogger(conf['log'])

    w2v = Word2Vec.load(os.path.join(conf['vocab_dir'], 'w2v_512')).wv
    c2v = Word2Vec.load(os.path.join(conf['vocab_dir'], 'c2v_512')).wv
    word_max_tokens = w2v.syn0.shape[0]
    word_embeddings = np.zeros((word_max_tokens + 1, conf['emb_size']), dtype='float32')
    word_embeddings[:w2v.syn0.shape[0]] = w2v.syn0
    code_max_tokens = c2v.syn0.shape[0]
    code_embeddings = np.zeros((code_max_tokens + 1, conf['emb_size']), dtype="float32")
    code_embeddings[:c2v.syn0.shape[0]] = c2v.syn0

    repo = pd.read_csv(conf['report'])
    if conf['bias'] is not None:
        used_commit_ids = repo[repo['is_bias'] == 'not localized']['commit_id'].tolist()
    else:
        used_commit_ids = repo['commit_id'].tolist()

    test_folds = conf['k_fold']
    test_folds = [i for i in range(4, 11)]  # use fold_4 - fold_10 as test
    avg_top1, avg_top5, avg_top10, avg_map, avg_mrr = 0, 0, 0, 0, 0
    for test_fold in test_folds:
        train_fold = [test_fold - 3, test_fold - 2, test_fold - 1]
        logger.info('training folds:{}, testing fold：{}'.format(train_fold, test_fold))
        pos_train = []
        neg_train = []
        k = 300  # negative samples number for each commit  # TODO
        for fold in train_fold:
            file = os.path.join(conf['data_dir'], 'fold_' + str(fold) + '.pkl')
            sub_train = pd.read_pickle(file)
            sub_train = sub_train[sub_train['commit_id'].isin(used_commit_ids)]
            sub_pos_train = sub_train[sub_train['label'] == 1]
            sub_neg_train = sub_train[sub_train['label'] == 0]
            rate = k * len(sub_pos_train) / len(sub_neg_train)
            sub_neg_train = sub_neg_train.sample(frac=rate)

            sub_pos_train['relevant_methods'] = \
                sub_pos_train.apply(lambda row: func(row['relevant_methods'],
                                                     sub_train, code_max_tokens, word_max_tokens), axis=1)

            sub_neg_train['relevant_methods'] = \
                sub_neg_train.apply(lambda row: func(row['relevant_methods'],
                                                     sub_train, code_max_tokens, word_max_tokens), axis=1)
            pos_train.append(sub_pos_train)
            neg_train.append(sub_neg_train)
            del sub_train, sub_pos_train, sub_neg_train
            gc.collect()

        pos_train = pd.concat(pos_train)
        neg_train = pd.concat(neg_train)
        pos_neg_rate = len(neg_train) // len(pos_train)
        pos_train_over_sampled = pd.DataFrame(np.repeat(pos_train.values, pos_neg_rate, axis=0),
                                              columns=pos_train.columns)
        train_data = pd.concat([pos_train_over_sampled, neg_train], axis=0).sample(frac=1).reset_index(drop=True)
        del pos_train, neg_train, pos_train_over_sampled
        gc.collect()

        logger.info('Training...')
        smnn, train_data = train_stage_1(conf, test_fold, train_data, repo)
        fcnn = train_stage_2(conf, test_fold, train_data)
        del train_data
        gc.collect()

        logger.info('Testing...')
        # load test set
        test_data = pd.read_pickle(os.path.join(conf['data_dir'], 'fold_' + str(test_fold) + '.pkl'))
        test_data = test_data[test_data['commit_id'].isin(used_commit_ids)]
        test_data['relevant_methods'] = test_data.apply(
            lambda row: func(row['relevant_methods'], test_data,
                             code_max_tokens, word_max_tokens), axis=1)

        test_data = shuffle(test_data)
        test_data = test_data.reset_index()
        commit_ids = test_data.drop_duplicates(subset=['commit_id'], keep='first')['commit_id'].to_list()
        logger.info('total bug report number:{}'.format(len(commit_ids)))
        test_result = test(conf, test_data, repo, smnn, fcnn)
        hit_1, hit_5, hit_10, map_value, mrr_value = evaluate(test_result)

        logger.info('test fold:{}, top1:{}, top5:{}, top10:{}, MAP:{}, MRR:{}'.format(test_fold, hit_1, hit_5, hit_10,
                                                                                      map_value, mrr_value))
        del test_data
        gc.collect()

        avg_top1 += hit_1
        avg_top5 += hit_5
        avg_top10 += hit_10
        avg_map += map_value
        avg_mrr += mrr_value

    avg_top1 /= len(test_folds)
    avg_top5 /= len(test_folds)
    avg_top10 /= len(test_folds)
    avg_map /= len(test_folds)
    avg_mrr /= len(test_folds)

    logger.info('with project prediction on {}, top1:{}, top5:{}, top10:{}, MAP:{}, MRR:{}'.
                format(conf['target'], avg_top1, avg_top5, avg_top10, avg_map, avg_mrr))


def cross_project_prediction(conf):
    logger = logging.getLogger(conf['log'])
    # load training data
    source_file = os.path.join(conf['report_dir'], conf['source'] + '.csv')
    target_file = os.path.join(conf['report_dir'], conf['target'] + '.csv')

    source_folds = os.path.join(os.path.join(conf['output_dir'], conf['source']), 'data/')
    target_folds = os.path.join(os.path.join(conf['output_dir'], conf['target']), 'data/')
    source_reports = pd.read_csv(source_file)
    target_reports = pd.read_csv(target_file)

    w2v = Word2Vec.load(os.path.join(conf['vocab_dir'], 'w2v_512')).wv
    c2v = Word2Vec.load(os.path.join(conf['vocab_dir'], 'c2v_512')).wv
    word_max_tokens = w2v.syn0.shape[0]
    word_embeddings = np.zeros((word_max_tokens + 1, conf['emb_size']), dtype='float32')
    word_embeddings[:w2v.syn0.shape[0]] = w2v.syn0
    code_max_tokens = c2v.syn0.shape[0]
    code_embeddings = np.zeros((code_max_tokens + 1, conf['emb_size']), dtype="float32")
    code_embeddings[:c2v.syn0.shape[0]] = c2v.syn0

    target_reports['report'] = target_reports.apply(lambda row: combine_smy_des(row['summary'], row['description']), axis=1)
    target_reports['report'] = word2id(w2v, target_reports['report'])

    if conf['bias'] is not None:
        used_commit_ids = target_reports[target_reports['is_bias'] == 'not localized']['commit_id'].tolist()
    else:
        used_commit_ids = target_reports['commit_id'].tolist()

    train_folds, test_folds = conf['k_fold'], conf['k_fold']
    pos_train = []
    neg_train = []
    k = 300
    for fold in range(1, train_folds + 1):
        file = os.path.join(source_folds, 'fold_' + str(fold) + '.pkl')
        sub_train = pd.read_pickle(file)
        sub_pos_train = sub_train[sub_train['label'] == 1]
        sub_neg_train = sub_train[sub_train['label'] == 0]
        rate = k * len(sub_pos_train) / len(sub_neg_train)
        sub_neg_train = sub_neg_train.sample(frac=rate)
        sub_pos_train['relevant_methods'] = \
            sub_pos_train.apply(lambda row: func(row['relevant_methods'],
                                                 sub_train, code_max_tokens, word_max_tokens), axis=1)
        sub_neg_train['relevant_methods'] = \
            sub_neg_train.apply(lambda row: func(row['relevant_methods'],
                                                 sub_train, code_max_tokens, word_max_tokens), axis=1)
        pos_train.append(sub_pos_train)
        neg_train.append(sub_neg_train)
        del sub_train, sub_pos_train, sub_neg_train
        gc.collect()

    pos_train = pd.concat(pos_train)
    neg_train = pd.concat(neg_train)
    pos_neg_rate = len(neg_train) // len(pos_train)
    pos_train_over_sampled = pd.DataFrame(np.repeat(pos_train.values, pos_neg_rate, axis=0), columns=pos_train.columns)
    train_data = pd.concat([pos_train_over_sampled, neg_train], axis=0).sample(frac=1).reset_index(drop=True)
    del pos_train, neg_train, pos_train_over_sampled
    gc.collect()

    # train
    smnn, train_data = train_stage_1(conf, 'cross', train_data, source_reports)
    fcnn = train_stage_2(conf, 'cross', train_data)
    del train_data
    gc.collect()

    avg_top1, avg_top5, avg_top10, avg_MAP, avg_MRR = 0, 0, 0, 0, 0
    for test_fold in range(1, test_folds + 1):
        # load test set
        test_data = pd.read_pickle(os.path.join(target_folds, 'fold_' + str(test_fold) + '.pkl'))
        test_data['relevant_methods'] = test_data.apply(
            lambda row: func(row['relevant_methods'], test_data,
                             code_max_tokens, word_max_tokens), axis=1)
        test_data = test_data[test_data['commit_id'].isin(used_commit_ids)]
        test_data = shuffle(test_data)
        test_data = test_data.reset_index()
        commit_ids = test_data.drop_duplicates(subset=['commit_id'], keep='first')['commit_id'].to_list()
        logger.info('total bug report number:{}'.format(len(commit_ids)))
        test_result = test(conf, test_data, target_reports, smnn, fcnn)
        top1, top5, top10, MAP, MRR = evaluate(test_result)

        del test_data
        gc.collect()

        avg_top1 += top1
        avg_top5 += top5
        avg_top10 += top10
        avg_MAP += MAP
        avg_MRR += MRR

    avg_top1 /= test_folds
    avg_top5 /= test_folds
    avg_top10 /= test_folds
    avg_MAP /= test_folds
    avg_MRR /= test_folds

    logger.info('corss project prediction (source:{} target:{}), '
                'top1:{}, top5:{}, top10:{}, MAP:{}, MRR:{}'.
                format(conf['source'], conf['target'], avg_top1, avg_top5, avg_top10, avg_MAP, avg_MRR))
