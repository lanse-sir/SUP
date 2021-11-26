import torch
import torch.nn as nn
from autocg.networks.attention import GlobalAttention
import math


class Pointer_gene(nn.Module):
    def __init__(self, vocab_size, emb_size, embedding, hidden_size, layers=1, dropout=0.0,
                 input_feeding=False, use_last_output=True, use_content=False):
        super(Pointer_gene, self).__init__()
        self.emb_size = emb_size
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.layers = layers
        self.input_feeding = input_feeding
        self.use_last_output = use_last_output
        self.dropout = nn.Dropout(dropout)
        self.input_size = self.emb_size + self.hidden_size if input_feeding else self.emb_size
        self.rnn = StackedGRU(self.input_size, self.hidden_size, layers, dropout)
        # it include two attention .
        self.tree_attention = GlobalAttention(self.hidden_size)
        self.sent_attention = GlobalAttention(self.hidden_size)

        # use content aware attention .
        self.use_content = use_content
        if self.use_content:
            print('Use content aware attention .')
            self.content_attention = GlobalAttention(self.hidden_size)
            self.ffn = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.vocab_size = vocab_size

        if self.use_last_output:
            self.soft_last_output = self.emb_size + self.hidden_size
        else:
            self.soft_last_output = self.hidden_size

        self.pgen = nn.Linear(self.hidden_size + self.hidden_size + self.emb_size, 1)
        self.gate = nn.Sigmoid()

        self.out = nn.Linear(self.soft_last_output, self.vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def gelu(x):
        """Implementation of the gelu activation function.
            For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
            Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def initial_output(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size, requires_grad=False)

    def forward(self, input_seq, pred_output, context, context_mask,
                sent_enc_outputs, sent_mask, enc_batch_extend_vocab, extra_oov,
                content_outputs=None, content_mask=None, hidden=None):
        # input_seq = input_dict['input_tensor'][:-1] # left shift.
        scores = []
        tree_att_w = []
        sent_att_w = []
        embeddings = self.embedding(input_seq)
        for emb in embeddings.split(1, dim=1):
            if self.input_feeding:
                input = torch.cat([emb.squeeze(1), pred_output], 1)
            else:
                input = emb.squeeze(1)

            rnn_output, hidden = self.rnn(input, hidden)

            tree_output, tree_att, tree_ctx = self.tree_attention(rnn_output, context, context_mask)
            tree_att_w.append(tree_att)

            output, sent_att, sent_ctx = self.sent_attention(tree_output, sent_enc_outputs, sent_mask)
            sent_att_w.append(sent_att)

            if self.use_content:
                con_attn_output, con_attn, con_ctx = self.content_attention(tree_output, content_outputs, content_mask)
                output = self.relu(self.ffn(output + con_attn_output))

            # compute copy probablity.
            pgen = self.gate(self.pgen(torch.cat((sent_ctx, output, emb.squeeze(1)), dim=1)))
            if self.use_last_output:
                pred_output = torch.cat([output, emb.squeeze(1)], 1)
            else:
                pred_output = torch.cat([output], 1)
            pred_output = self.dropout(pred_output)
            logits = self.out(pred_output)
            voc_score = self.softmax(logits)
            # pointer generator section.
            if extra_oov.size(1) != 0:
                voc_score = torch.cat((voc_score, extra_oov), dim=1)
            voc_score = voc_score * pgen
            attn_dist = (1.0 - pgen) * sent_att
            voc_score.scatter_add_(1, enc_batch_extend_vocab, attn_dist)
            voc_score = torch.log(voc_score + 10e-6)
            scores.append(voc_score)
        return torch.stack(scores), hidden, output, torch.stack(tree_att_w), torch.stack(sent_att_w)


class StackedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, layer, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = layer
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_1 = []
        for i, layer in enumerate(self.layers):
            if hidden is not None:
                h_1_i = layer(input, hidden[i])
            else:
                h_1_i = layer(input, hidden)

            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
        h_1 = torch.stack(h_1)
        return input, h_1
