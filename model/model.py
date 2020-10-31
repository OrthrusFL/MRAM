import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SMNN(nn.Module):
    def __init__(self, code_vocab_size, word_vocab_size, pretrained_code_embeddings, pretrained_word_embeddings,
                 embedding_dim,
                 use_gpu):
        super(SMNN, self).__init__()
        self.batch_size = None
        self.use_gpu = use_gpu
        self.augment = True
        self.word_vocab_size = word_vocab_size
        self.code_vocab_size = code_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = 256
        self.n_layers = 1
        self.augment_threshold = 5
        self.CODE_PADDED = code_vocab_size - 1
        self.WORD_PADDED = word_vocab_size - 1
        self.word_embeddings = nn.Embedding(self.word_vocab_size, self.embedding_dim, padding_idx=self.WORD_PADDED)
        self.code_embeddings = nn.Embedding(self.code_vocab_size, self.embedding_dim, padding_idx=self.CODE_PADDED)
        self.token_encoder = RNNEncoder(embedding_dim)
        self.api_encoder = RNNEncoder(embedding_dim)
        self.comment_encoder = RNNEncoder(embedding_dim)
        self.report_encoder = RNNEncoder(embedding_dim)
        self.fcnn = FLNN()
        self.similarity_calculator = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        if pretrained_word_embeddings is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_word_embeddings))

        if pretrained_code_embeddings is not None:
            self.code_embeddings.weight.data.copy_(torch.from_numpy(pretrained_code_embeddings))

    def encode(self, inputs, label):
        if label == 1:
            inputs = self.word_embeddings(inputs)
        else:
            inputs = self.code_embeddings(inputs)
        return inputs

    def embed_method(self, report_vector, token, token_len, api, api_len, comment, comment_len):
        batch_size = token.size()[0]
        token = self.token_encoder(self.encode(token, 0), token_len).unsqueeze(1)
        api = self.api_encoder(self.encode(api, 0), api_len).unsqueeze(1)
        comment = self.comment_encoder(self.encode(comment, 1), comment_len).unsqueeze(1)
        # semantic capture by attention
        method = torch.cat((token, api, comment), 1)

        if self.use_gpu:
            method = method.cuda()
        attn = F.softmax(torch.bmm(report_vector, method.view(batch_size, self.embedding_dim, 3)), dim=0)
        method = torch.bmm(attn, method).squeeze(1)
        return method

    def augment_short_method(self, report, short_emb, batch_rel_methods):
        retrieved_info = torch.Tensor(size=short_emb[0].size())
        if self.use_gpu:
            retrieved_info = retrieved_info.cuda()

        for i in range(self.batch_size):
            rel_methods = batch_rel_methods[i]
            length = len(rel_methods['token'])
            if length > 0:  # is short method
                # embed relevant methods
                rel_token = rel_methods['token']
                rel_token_len = rel_methods['token_len']
                rel_api = rel_methods['api']
                rel_api_len = rel_methods['api_len']
                rel_comment = rel_methods['comment']
                rel_comment_len = rel_methods['comment_len']
                if self.use_gpu:
                    rel_token, rel_token_len, rel_api, rel_api_len, rel_comment, rel_comment_len = \
                        rel_token.cuda(), rel_token_len.cuda(), rel_api.cuda(), rel_api_len.cuda(), \
                        rel_comment.cuda(), rel_comment_len.cuda()

                rel_reports = report[i].unsqueeze(0).repeat(length, 1, 1)

                rel_emb = self.embed_method\
                    (rel_reports, rel_token, rel_token_len, rel_api, rel_api_len, rel_comment, rel_comment_len)

                rel_emb = rel_emb.unsqueeze(-1)
                short_emb_repeat = short_emb[i].unsqueeze(0).repeat(length, 1, 1)
                attn = F.softmax(torch.bmm(short_emb_repeat, rel_emb), dim=0)
                expand_emb = sum(torch.bmm(attn, short_emb_repeat).squeeze(1))
                expand_emb = expand_emb.unsqueeze(0)
            else:  # not short method
                expand_emb = short_emb[i].unsqueeze(0)
            if i == 0:
                retrieved_info = expand_emb
            else:
                retrieved_info = torch.cat([retrieved_info, expand_emb], dim=0)

        input_size = short_emb[0].size()[0]
        hidden_size = input_size

        gru_cell = nn.GRUCell(input_size, hidden_size)  # bigru 所以需要*2
        if self.use_gpu:
            gru_cell = gru_cell.cuda()

        augmented_emb = gru_cell(short_emb, retrieved_info)
        return augmented_emb

    def forward(self, report, report_len, token, token_len, api, api_len, comment, comment_len, batch_rel_methods):
        self.batch_size = report.size()[0]

        # Semantic matching
        report = self.report_encoder(self.encode(report, 1), report_len).unsqueeze(1)
        method = self.embed_method(report, token, token_len, api, api_len, comment, comment_len)
        report = report.squeeze()

        # augment short methods

        if self.augment:
            method = self.augment_short_method(report, method, batch_rel_methods)

        # similarity = F.cosine_similarity(method, report.squeeze()).unsqueeze(1)
        feature = torch.cat((method, report), 1)
        if self.use_gpu:
            feature = feature.cuda()
        similarity = self.similarity_calculator(feature)
        return similarity


class FLNN(nn.Module):
    def __init__(self):
        super(FLNN, self).__init__()
        self.feature_combinator = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, features):
        faulty_prob = self.feature_combinator(features)
        return faulty_prob


class RNNEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(RNNEncoder, self).__init__()
        self.hidden_size = 256
        self.n_layers = 1
        self.batch_size = None
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, batch_first=True, bidirectional=True)

    def forward(self, inputs, input_lens):
        self.batch_size = inputs.size()[0]
        # sort and pack sequence
        input_lens_sorted, indices = input_lens.sort(descending=True)
        inputs_sorted = inputs.index_select(0, indices)
        inputs = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)

        # lstm
        output, (h_n, c_n) = self.lstm(inputs)  # hids:[batch_size, seq_len, hid_sz*2](bi-lstm,hid_size*2)
        # reorder and pad
        # _, inv_indices = indices.sort()
        # output, lens = pad_packed_sequence(output, batch_first=True)
        # output = F.dropout(output, p=0.25, training=self.training)
        # output = output.index_select(0, inv_indices)
        # pooled_encoding = F.max_pool1d(output.transpose(1, 2), seq_len).squeeze(2)  # [batch_size x hid_size*2]
        # encoding = torch.tanh(pooled_encoding)

        _, inv_indices = indices.sort()
        hids, lens = pad_packed_sequence(output, batch_first=True)
        # hids = F.dropout(hids, p=0.25, training=self.training)
        # hids = hids.index_select(0, inv_indices)
        h_n = h_n.index_select(1, inv_indices)
        h_n = h_n.view(self.n_layers, 2, self.batch_size, self.hidden_size)  # [n_layers x n_dirs x batch_sz x hid_sz]
        h_n = h_n[-1]
        encoding = h_n.view(self.batch_size, -1)  # [batch_sz x (n_dirs*hid_sz)]
        return encoding
