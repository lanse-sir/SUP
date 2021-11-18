import os
import torch
import torch.nn as nn
from autocg.encoder.rnn_encoder import RNNEncoder
from autocg.networks.embeddings import Embeddings

class binary_classifer(nn.Module):
    def __init__(self, args, vocab, word_embedding=None):
        super(binary_classifer, self).__init__()
        self.vocab_size = len(vocab)
        self.word_vocab = vocab
        self.label_num = args.label_num
        self.args = args
        self.word_emb = Embeddings(len(self.word_vocab), args.enc_embed_dim, args.enc_ed, add_position_embedding=False,
                                   padding_idx=0)
        if word_embedding is not None:
            self.word_emb.weight.data.copy_(torch.from_numpy(word_embedding))
        # encoder .
        self.encoder = RNNEncoder(
            vocab_size=self.vocab_size,
            max_len=args.src_max_time_step,
            input_size=args.enc_embed_dim,
            hidden_size=args.enc_hidden_dim,
            embed_droprate=args.enc_ed,
            rnn_droprate=args.enc_rd,
            n_layers=args.enc_num_layers,
            bidirectional=args.bidirectional,
            rnn_cell=args.rnn_type,
            variable_lengths=True,
            embedding=self.word_emb
        )

        self.enc_factor = 2 if args.bidirectional else 1
        self.enc_dim = args.enc_hidden_dim * self.enc_factor
        # classification .
        self.fc = nn.Linear(self.enc_dim, self.label_num)
        # loss function .
        self.loss = nn.CrossEntropyLoss()
        
        
    def forward(self, input):
        src_var = input['src tensor']
        src_length = input['src lengths']
        target = input['target']
        encoder_outputs, encoder_hidden = self.encoder.forward(input_var=src_var, input_lengths=src_length)
        hn = encoder_hidden[0]
        encoder_hidden = torch.cat((hn[0], hn[1]), dim=-1)
        logits = self.fc(encoder_hidden)
        loss = self.loss(logits, target)
        return loss

    def predict(self, input):
        src_var = input['src tensor']
        src_length = input['src lengths']
        seq_recover = input['seq recover']
        encoder_outputs, encoder_hidden = self.encoder.forward(input_var=src_var, input_lengths=src_length)
        hn = encoder_hidden[0]
        encoder_hidden = torch.cat((hn[0], hn[1]), dim=-1)
        encoder_hidden = encoder_hidden[seq_recover]
        logits = self.fc(encoder_hidden)
        return logits

    def save(self, path):
        dir_name = os.path.dirname(path)
        # remove file name, return directory.
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'args': self.args,
            'word_vocab': self.word_vocab,
            'state_dict': self.state_dict(),
        }
        torch.save(params, path)
