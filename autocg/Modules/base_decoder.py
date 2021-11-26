import torch
import torch.nn as nn


class BASEDecoder(nn.Module):
    def __init__(self, vocab_size, emb_size, embedding, hidden_size, layers=1, dropout=0.0,
                 input_feeding=False, use_last_output=False):
        super(BASEDecoder, self).__init__()
        self.emb_size = emb_size
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.input_feeding = input_feeding
        self.use_last_output = use_last_output
        self.dropout = nn.Dropout(dropout)
        self.input_size = self.emb_size + self.hidden_size if input_feeding else self.emb_size
        self.rnn = StackedGRU(self.input_size, self.hidden_size, layers, dropout)
        self.vocab_size = vocab_size

        if self.use_last_output:
            self.soft_last_output = self.emb_size + self.hidden_size
        else:
            self.soft_last_output = self.hidden_size

        self.out = nn.Linear(self.soft_last_output, self.vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def initial_output(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size, requires_grad=False)

    def forward(self, input_seq, prev_output, hidden=None):
        # input_seq = input_dict['input_tensor'][:-1] # left shift.
        output = prev_output
        scores = []

        embeddings = self.embedding(input_seq)
        for emb in embeddings.split(1, dim=1):
            if self.input_feeding:
                input = torch.cat([emb.squeeze(1), output], 1)
            else:
                input = emb.squeeze(1)

            output, hidden = self.rnn(input, hidden)
            if self.use_last_output:
                pred_output = torch.cat([output, emb.squeeze(1)], 1)
            else:
                pred_output = output
            pred_output = self.dropout(pred_output)
            logits = self.out(pred_output)
            score = self.logsoftmax(logits)
            scores.append(score)
        return torch.stack(scores), hidden, output


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
