import os
import logging
import torch
import numpy as np
import torch.optim as optim
from model import SMNN, FLNN
from sklearn.utils import shuffle
from gensim.models.word2vec import Word2Vec
from prepare import word2id
from utils import combine_smy_des


def pad_sequence(seqs, pad_index):
    lens = [len(item) for item in seqs]
    max_len = max(lens)
    padded_seqs = []
    size = len(seqs)
    for i in range(size):
        for j in range(lens[i]):
            padded_seqs.append(seqs[i][j])
        diff = max_len - lens[i]
        for x in range(diff):
            padded_seqs.append(pad_index)

    padded_seqs = torch.autograd.Variable(torch.LongTensor(padded_seqs))
    padded_seqs = padded_seqs.view(size, max_len)  # batch_size, max_len
    # padded_seqs = padded_seqs.view(size, 1, max_len)  # batch_size, 1, max_len
    return padded_seqs


def get_batch_stage_1(data, idx, batch, repo, code_pad_idx, word_pad_idx, max_len):
    sub_data = data[idx:idx + batch]
    tokens, reports, apis, comments, rel_methods, labels = [], [], [], [], [], []

    for index, row in sub_data.iterrows():
        commit_id = row['commit_id']
        token = row['token']
        report = repo[repo['commit_id'] == commit_id]['report'].values[0]
        api = row['api']
        comment = row['comment']
        token = token[:max_len]
        report = report[:max_len]
        # num_statements = row['statements']
        rel = row['relevant_methods']

        reports.append(report)
        tokens.append(token)
        apis.append(api + [code_pad_idx])
        comments.append(comment + [word_pad_idx])
        labels.append(row['label'])

        if len(rel['token']) > 0:
            rel['token_len'] = torch.FloatTensor([len(i) for i in rel['token']])
            rel['api_len'] = torch.FloatTensor([len(i) for i in rel['api']])
            rel['comment_len'] = torch.FloatTensor([len(i) for i in rel['comment']])
            rel['token'] = pad_sequence(rel['token'], code_pad_idx)
            rel['api'] = pad_sequence(rel['api'], code_pad_idx)
            rel['comment'] = pad_sequence(rel['comment'], word_pad_idx)

        rel_methods.append(rel)

    report_lens = torch.FloatTensor([len(i) for i in reports])
    token_lens = torch.FloatTensor([len(i) for i in tokens])
    api_lens = torch.FloatTensor([len(i) for i in apis])
    comment_lens = torch.FloatTensor([len(i) for i in comments])

    reports = pad_sequence(reports, word_pad_idx)
    tokens = pad_sequence(tokens, code_pad_idx)
    apis = pad_sequence(apis, code_pad_idx)
    comments = pad_sequence(comments, word_pad_idx)

    return reports, report_lens, tokens, token_lens, apis, api_lens, comments, comment_lens, rel_methods, torch.FloatTensor(labels)


def get_batch_stage_2(data, idx, batch):
    sub_data = data[idx:idx + batch]
    features, labels = [], []
    for index, row in sub_data.iterrows():
        features.append([row['similarity'], row['cfs'], row['bfr'], row['bff']])
        if row['label'] == 1:
            labels.append([0, 1])
        elif row['label'] == 0:
            labels.append([1, 0])
    return torch.FloatTensor(features), torch.FloatTensor(labels)


def train_stage_1(conf, test_fold, train_data, reports):
    logger = logging.getLogger(conf['log'])

    # training parameters
    n_epochs = conf['epoch']
    batch_size = conf['batch_size']
    emb_size = conf['emb_size']
    if conf['gpu'] is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = conf['gpu']
        use_gpu = True
    else:
        use_gpu = False

    w2v = Word2Vec.load(os.path.join(conf['vocab_dir'], 'w2v_512')).wv
    c2v = Word2Vec.load(os.path.join(conf['vocab_dir'], 'c2v_512')).wv
    word_max_tokens = w2v.syn0.shape[0]
    word_embeddings = np.zeros((word_max_tokens + 1, emb_size), dtype='float32')
    word_embeddings[:w2v.syn0.shape[0]] = w2v.syn0
    code_max_tokens = c2v.syn0.shape[0]
    code_embeddings = np.zeros((code_max_tokens + 1, emb_size), dtype="float32")
    code_embeddings[:c2v.syn0.shape[0]] = c2v.syn0

    reports['report'] = reports.apply(lambda row: combine_smy_des(row['summary'], row['description']), axis=1)
    reports['report'] = word2id(w2v, reports['report'])

    # load model
    model = SMNN(code_vocab_size=code_max_tokens + 1,
                 word_vocab_size=word_max_tokens + 1,
                 pretrained_code_embeddings=code_embeddings,
                 pretrained_word_embeddings=word_embeddings,
                 embedding_dim=emb_size,
                 use_gpu=use_gpu)

    train_len = len(train_data)
    loss_similarity = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if use_gpu:
        loss_similarity = loss_similarity.cuda()
        model= model.cuda()

    for epoch in range(1, n_epochs + 1):
        train_data = shuffle(train_data)
        i = 0
        total_loss_similarity = 0
        logger.info('training epoch:{}'.format(epoch))
        while i < train_len:
            report, report_len, token, token_len, api, api_len, comment, comment_len, rel_methods, label = \
                get_batch_stage_1(train_data, i, batch_size, reports, code_max_tokens, word_max_tokens, conf['max_len'])

            if use_gpu:
                report, report_len, token, token_len, api, api_len, comment, comment_len, label = \
                    report.cuda(), report_len.cuda(), token.cuda(), token_len.cuda(), \
                    api.cuda(), api_len.cuda(), comment.cuda(), comment_len.cuda(), label.cuda()

            model.train()
            model.zero_grad()
            similarity_output = model(report, report_len, token, token_len, api, api_len, comment, comment_len, rel_methods)
            loss = loss_similarity(similarity_output, label)
            loss.backward()
            optimizer.step()
            total_loss_similarity += loss
            i += batch_size
        logger.info(
            'epoch: {}:, similarity loss:{}'.format(epoch, total_loss_similarity))

    if test_fold:
        model_save_path = conf['model_dir'] + 'smnn_epoch_' + str(test_fold) +'.pth'
        torch.save(model.state_dict(), model_save_path)

    i = 0
    predict_similarity = []
    while i < train_len:
        report, report_len, token, token_len, api, api_len, comment, comment_len, rel_methods, label = \
            get_batch_stage_1(train_data, i, batch_size, reports, code_max_tokens, word_max_tokens,
                              conf['max_len'])

        if use_gpu:
            report, report_len, token, token_len, api, api_len, comment, comment_len, label = \
                report.cuda(), report_len.cuda(), token.cuda(), token_len.cuda(), \
                api.cuda(), api_len.cuda(), comment.cuda(), comment_len.cuda(), label.cuda()

        with torch.no_grad():
            model.eval()
            similarity = model(report, report_len, token, token_len, api, api_len, comment, comment_len,rel_methods)
        i += batch_size

        similarity = similarity.cpu().numpy().tolist()
        for s in similarity:
            predict_similarity.append(s[0])
    train_data['similarity'] = predict_similarity
    return model, train_data


def train_stage_2(conf, test_fold, train_data,):
    logger = logging.getLogger(conf['log'])
    n_epochs = conf['epoch']
    batch_size = conf['batch_size']
    train_len = len(train_data)
    if conf['gpu'] is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = conf['gpu']
        use_gpu = True
    else:
        use_gpu = False

    model = FLNN()
    loss_fault = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    if use_gpu:
        model = model.cuda()
        loss_fault = loss_fault.cuda()

    for epoch in range(1, n_epochs + 1):
        total_loss_fault = 0
        train_data = shuffle(train_data)
        i = 0
        while i < train_len:
            features, label = get_batch_stage_2(train_data, i, batch_size)

            if use_gpu:
                features, label = features.cuda(), label.cuda()

            model.train()
            model.zero_grad()
            fault_output = model(features)
            loss = loss_fault(fault_output, label)
            loss.backward()
            optimizer.step()
            total_loss_fault += loss
            i += batch_size
        logger.info('epoch: {}:, total loss:{}'.format(epoch, total_loss_fault))
        if test_fold:
            model_save_path = conf['model_dir'] + 'fcnn_epoch_' + str(test_fold) + '.pth'
            torch.save(model.state_dict(), model_save_path)

    return model


def test(conf, test_data, reports, smnn, fcnn):
    logger = logging.getLogger(conf['log'])

    batch_size = conf['batch_size']
    emb_size = conf['emb_size']
    test_len = len(test_data)

    w2v = Word2Vec.load(os.path.join(conf['vocab_dir'], 'w2v_512')).wv
    c2v = Word2Vec.load(os.path.join(conf['vocab_dir'], 'c2v_512')).wv
    word_max_tokens = w2v.syn0.shape[0]
    word_embeddings = np.zeros((word_max_tokens + 1, emb_size), dtype='float32')
    word_embeddings[:w2v.syn0.shape[0]] = w2v.syn0
    code_max_tokens = c2v.syn0.shape[0]
    code_embeddings = np.zeros((code_max_tokens + 1, emb_size), dtype="float32")
    code_embeddings[:c2v.syn0.shape[0]] = c2v.syn0

    if conf['gpu'] is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = conf['gpu']
        use_gpu = True
    else:
        use_gpu = False

    predict_similarity = []
    j = 0
    while j < test_len:
        report, report_len, token, token_len, api, api_len, comment, comment_len,  rel_methods, _ = \
            get_batch_stage_1(test_data, j, batch_size, reports, code_max_tokens, word_max_tokens,
                              conf['max_len'])

        if use_gpu:
            report, report_len, token, token_len, api, api_len, comment, comment_len = \
                report.cuda(), report_len.cuda(), token.cuda(), token_len.cuda(), \
                api.cuda(), api_len.cuda(), comment.cuda(), comment_len.cuda()
        with torch.no_grad():
            smnn.eval()
            similarity = smnn(report, report_len, token, token_len, api, api_len, comment, comment_len,rel_methods)
            similarity = similarity.cpu().numpy().tolist()
            for s in similarity:
                predict_similarity.append(s[0])

        j += batch_size
    test_data['similarity'] = predict_similarity

    predict_faulty_prob = []
    j = 0
    test_len = len(test_data)
    while j < test_len:
        features, label = get_batch_stage_2(test_data, j, batch_size)
        if use_gpu:
            features, label = features.cuda(), label.cuda()
        with torch.no_grad():
            fcnn.eval()
            fault_prob = fcnn(features)
            fault_prob = fault_prob[:, -1]
        predict_faulty_prob += fault_prob.cpu().numpy().tolist()
        j += batch_size
    test_data['fault_prob'] = predict_faulty_prob

    return test_data

