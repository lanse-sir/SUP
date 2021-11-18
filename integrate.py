import sys
sys.path.append('../')
import os
import torch
import torch.nn as nn
from cvae_model.model_cvae import model_cvae
from cvae_model.classifer.model_seq2seq import seq2seq
from cvae_model.train_cvae import load_vocab
from autocg.pretrain_embedding import load_embedding
from autocg.networks.embeddings import Embeddings

class INTE_Model(nn.Module):
    def __init__(self, args, funetine_config):
        super(INTE_Model, self).__init__()
        if args.finetune:
            self.cvae_params = torch.load(args.cvae_file)
            if funetine_config:
                print('Using fine-tune config......')
                self.cvae = model_cvae(args, self.cvae_params['word_vocab'],
                                       self.cvae_params['parse_vocab'])
            else:
                print('Using load cvae model config......')
                self.cvae = model_cvae(self.cvae_params['args'], self.cvae_params['word_vocab'],
                                       self.cvae_params['parse_vocab'])
            
            self.word_vocab = self.cvae_params['word_vocab']
            self.cvae.load_state_dict(self.cvae_params['state_dict'])
        else:
            word_vocab = load_vocab(args.vocab)
            parse_vocab = load_vocab(args.parser_vocab)
            print('Training from scratch ....................................................')
            self.word_vocab = word_vocab
            # load pretrained word embedding .
            if os.path.exists(args.pretrain_emb) and args.debug is not True:
                word_embedding = load_embedding(args.pretrain_emb, word_vocab)
            else:
                word_embedding = None
            self.cvae = model_cvae(args, word_vocab, parse_vocab, word_embedding=word_embedding)

        # syntactic classifer.
        self.params = torch.load(args.style_transfer_file, map_location=lambda storage, loc: storage)
        self.classifer = seq2seq(self.params['args'], self.params['word_vocab'], self.params['parse_vocab'])

        self.classifer.load_state_dict(self.params['state_dict'])
        for p in self.classifer.parameters():
            p.requires_grad = False

        # semantic similarity .
        #sim_word_embedding = load_embedding(args.sim_emb, self.word_vocab)
        #self.sim_word_emb = Embeddings(len(self.word_vocab), args.enc_embed_dim, args.enc_ed, add_position_embedding=False,
        #                           padding_idx=0)
        #self.sim_word_emb.weight.data.copy_(torch.from_numpy(sim_word_embedding))
        #for p in self.sim_word_emb.parameters():
            #p.requires_grad = False