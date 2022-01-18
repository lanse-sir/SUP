import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class classifier(nn.Module):
    def __init__(self, args, vocab, word_embedding=None):
        super(classifier, self).__init__()
        self.args = args
        self.label_num = args.label_num
        self.label_emb_dim = args.label_emb_dim
        self.embedding_dim = args.word_emb_dim
        self.vocab = vocab
        self.vocab_num = len(vocab)
        self.label_embedding = nn.Embedding(1, self.label_emb_dim)
        self.word_embedding = nn.Embedding(self.vocab_num, self.embedding_dim, padding_idx=0)
        self.hidden_size = args.hidden_size
        self.layers = args.encoder_layer
        if word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(word_embedding))
        self.rnn = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size, batch_first=True, bidirectional=True,
                           num_layers=self.layers)

        self.att_w = nn.Linear(self.hidden_size * 2, self.label_emb_dim)
        self.cf_w = nn.Linear(self.hidden_size * 2, self.label_num)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, src_seq_tensor, src_seq_mask, label_seq_tensor=None, src_seq_length=None):
        batch_size = src_seq_tensor.size(0)
        seq_length = src_seq_tensor.size(1)
        word_embedded = self.word_embedding(src_seq_tensor)
        if src_seq_length is not None:
            word_embedded = nn.utils.rnn.pack_padded_sequence(word_embedded, src_seq_length, batch_first=True)
        output, hidden = self.rnn(word_embedded)
        if src_seq_length is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # compute attention scores .
        sent_type_tensor = torch.LongTensor([[0] for i in range(batch_size)])
        if torch.cuda.is_available():
            sent_type_tensor = sent_type_tensor.cuda()
        sent_type_emb = self.label_embedding(sent_type_tensor)
        att_hidden = self.att_w(output.contiguous().view(-1, self.hidden_size * 2)).view(batch_size, seq_length, -1).permute(0, 2, 1)
        att_scores = torch.matmul(sent_type_emb, att_hidden).squeeze(1)
        if src_seq_mask is not None:
            att_scores = att_scores.masked_fill(1 - src_seq_mask, -np.inf)
        sm_att = F.softmax(att_scores, dim=-1)
        weight_sent_hidden = torch.matmul(sm_att.unsqueeze(1), output).squeeze(1)
        logit = self.cf_w(weight_sent_hidden)
        loss = None
        if label_seq_tensor is not None:
            loss = self.loss_function(logit, label_seq_tensor)
        return loss, logit, sm_att


    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        params = {
            'args':self.args,
            'vocab':self.vocab,
            'state_dict':self.state_dict()
        }
        torch.save(params, path)