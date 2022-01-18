import os
import torch
import torch.nn as nn
from autocg.encoder.rnn_encoder import RNNEncoder
from autocg.decoder.rnn_decoder import RNNDecoder
from autocg.networks.embeddings import Embeddings
from autocg.networks.bridger import MLPBridger
from autocg.networks.bridger import MultiTensorFusion
from autocg.decoder.beam_decoder import TopKDecoder
from autocg.utils.nn_funcs import dict_id2word
# from autocg.networks.attention import Attention_two_encoder
import numpy as np


class atcg(nn.Module):
    def __init__(self, args, word_vocab, parse_vocab, word_embedding=None):
        super(atcg, self).__init__()
        self.args = args
        self.word_vocab = word_vocab
        self.parse_vocab = parse_vocab
        self.dropout_r = args.unk_rate
        self.sos_id = word_vocab['<s>']
        self.eos_id = word_vocab['</s>']
        self.pad_id = word_vocab['<PAD>']
        self.unk_id = word_vocab['<unk>']
        # vae training ...
        self.k = args.k
        self.x0 = args.x0
        self.step_kl_weight = args.init_step_kl_weight
        # self.word_emb = word_emb
        # self.parse_emb = parse_emb
        # self.num_sent_pattern = 4
        self.sent_pattern_emb_dim = args.enc_embed_dim
        self.word_emb = Embeddings(len(self.word_vocab), args.enc_embed_dim, args.enc_ed, add_position_embedding=False,
                                   padding_idx=0)
        if word_embedding is not None:
            self.word_emb.weight.data.copy_(torch.from_numpy(word_embedding))
        self.parse_emb = Embeddings(len(self.parse_vocab), args.enc_embed_dim, args.enc_ed,
                                    add_position_embedding=False, padding_idx=0)

        # self.sent_pattern_emb = nn.Embedding(self.num_sent_pattern, self.sent_pattern_emb_dim)
        # bidirectional ?
        self.enc_factor = 2 if args.bidirectional else 1
        self.enc_hidden_size = args.enc_hidden_dim
        self.enc_dim = args.enc_hidden_dim * self.enc_factor

        if args.mapper_type == "link":
            self.dec_hidden = self.enc_dim
        elif args.use_attention:
            self.dec_hidden = self.enc_dim
        else:
            self.dec_hidden = args.dec_hidden_dim

        self.sent_encoder = RNNEncoder(
            vocab_size=len(self.word_vocab),
            max_len=args.sent_max_time_step,
            input_size=args.enc_embed_dim,
            hidden_size=args.enc_hidden_dim,
            embed_droprate=args.enc_ed,
            rnn_droprate=args.enc_rd,
            n_layers=args.enc_num_layers,
            bidirectional=args.bidirectional,
            rnn_cell=args.rnn_type,
            variable_lengths=True,
            embedding=self.word_emb)

        self.parse_encoder = RNNEncoder(
            vocab_size=len(self.word_vocab),
            max_len=args.sent_max_time_step,
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
        # text generation base on language template .
        # if args.use_attention:
        #     self.att = Attention_two_encoder(self.dec_hidden)
        # else:
        #     self.att = None

        self.sent_decoder = RNNDecoder(
            vocab=len(self.word_vocab),
            max_len=args.sent_max_time_step,
            input_size=args.dec_embed_dim,
            hidden_size=self.dec_hidden,
            embed_droprate=args.dec_ed,
            rnn_droprate=args.dec_rd,
            word_dropout=self.dropout_r,
            n_layers=args.dec_num_layers,
            rnn_cell=args.rnn_type,
            use_attention=args.use_attention,
            embedding=self.word_emb,
            eos_id=self.word_vocab['</s>'],
            sos_id=self.word_vocab['<s>'],
            unk_id = self.word_vocab['<unk>']

        )
        self.parse_decoder = RNNDecoder(
            vocab=len(self.parse_vocab),
            max_len=args.parse_max_time_step,
            input_size=args.dec_embed_dim,
            hidden_size=self.dec_hidden,
            embed_droprate=args.dec_ed,
            rnn_droprate=args.dec_rd,
            word_dropout=self.dropout_r,
            n_layers=args.dec_num_layers,
            rnn_cell=args.rnn_type,
            use_attention=args.use_attention,
            embedding=self.parse_emb,
            eos_id=self.parse_vocab['</s>'],
            sos_id=self.parse_vocab['<s>'],
            unk_id=self.parse_vocab['<unk>']
        )

        self.sent_bridge = MLPBridger(
            rnn_type=args.rnn_type,
            mapper_type=args.mapper_type,
            encoder_dim=self.enc_dim,
            encoder_layer=args.enc_num_layers,
            decoder_dim=self.dec_hidden,
            decoder_layer=args.dec_num_layers
        )

        self.parse_bridge = MLPBridger(
            rnn_type=args.rnn_type,
            mapper_type=args.mapper_type,
            encoder_dim=self.enc_dim,
            encoder_layer=args.enc_num_layers,
            decoder_dim=self.dec_hidden,
            decoder_layer=args.dec_num_layers
        )
        # VAE part .
        self.latent_dim = args.latent_size
        self.sem_mean = nn.Linear(self.enc_hidden_size * self.enc_factor, self.latent_dim)
        self.sem_logv = nn.Linear(self.enc_hidden_size * self.enc_factor, self.latent_dim)
        self.sem_to_hidden = nn.Linear(self.latent_dim, self.enc_hidden_size)

        self.syn_mean = nn.Linear(self.enc_hidden_size * self.enc_factor, self.latent_dim)
        self.syn_logv = nn.Linear(self.enc_hidden_size * self.enc_factor, self.latent_dim)
        self.syn_to_hidden = nn.Linear(self.latent_dim, self.enc_hidden_size)

        # hidden state fusion .
        self.hidden_f = nn.Linear(self.dec_hidden * 2, self.dec_hidden)

        self.sent_beam_search = TopKDecoder(self.sent_decoder, args.sample_size)
        self.parse_beam_search = TopKDecoder(self.parse_decoder, args.sample_size)

        # syntax and label carry out fusion.
        self.sent_tensor_fusion = MultiTensorFusion(mapping_type=args.mapper_type,
                                                    encoder_dim=self.enc_dim,
                                                    encoder_layer=args.enc_num_layers,
                                                    decoder_dim=self.dec_hidden,
                                                    decoder_layer=args.dec_num_layers)

        print("enc layer: {}, dec layer: {}, type: {}, with attention: {}, word dropout: {}".format(
            args.enc_num_layers,
            args.dec_num_layers,
            args.rnn_type,
            args.use_attention,
            self.dropout_r))

    def sem_hidden_to_latent(self, hidden, sample=False):
        batch_size = hidden.size(1)
        hidden_dim = hidden.size(2)
        # hidden = hidden.permute(1, 0, 2).contiguous()
        # if self.enc_factor > 1 :
        hidden = hidden.view(batch_size, hidden_dim * self.enc_factor)
        mean = self.sem_mean(hidden)
        logv = self.sem_logv(hidden)
        if sample:
            std = torch.exp(0.5 * logv)
            z = torch.randn([batch_size, self.latent_dim]).cuda()
            # have two layer lstm .
            # z = z.repeat(1, 2).view(batch_size, -1, self.latent_dim)
            z = z * std + mean
        else:
            z = mean
        return z, mean, logv

    def syn_hidden_to_latent(self, hidden, sample=False):
        # input : (bidirection*layer, batch, dim)
        # output : (bidirection*layer, batch, dim)
        batch_size = hidden.size(1)
        hidden_dim = hidden.size(2)
        # hidden = hidden.permute(1, 0, 2).contiguous()
        # if self.enc_factor > 1 :
        # hidden = hidden.view(batch_size, -1, hidden_dim * self.enc_factor)
        mean = self.syn_mean(hidden.view(-1, hidden_dim)).view(-1, batch_size, self.latent_dim)
        logv = self.syn_logv(hidden.view(-1, hidden_dim)).view(-1, batch_size, self.latent_dim)
        if sample:
            std = torch.exp(0.5 * logv).cuda()
            z = torch.randn(mean.size()).cuda()
            # have two layer lstm .
            # z = z.repeat(1, 2).view(batch_size, -1, self.latent_dim)
            z = z * std + mean
        else:
            z = mean
        return z, mean, logv

    def latent_to_hidden(self, function, latent):
        batch_size = latent.size(0)
        hidden = function(latent)
        # print(hidden.size())
        hidden = hidden.view(1, batch_size, self.enc_hidden_size)
        # hidden = hidden.permute(1, 0, 2).contiguous()
        return hidden

    def hidden_fusion(self, syn_hidden, sent_hidden):
        batch_size = syn_hidden.size(1)
        hidden_cat = torch.cat([sent_hidden, syn_hidden], dim=2).view(-1, self.dec_hidden * 2)
        map_hidden = self.hidden_f(hidden_cat).view(-1, batch_size, self.dec_hidden)
        return map_hidden

    def kl_anneal_function(self, anneal_function, step):
        if anneal_function == "fixed":
            return 1.0
        elif anneal_function == 'logistic':
            return float(1 / (1 + np.exp(-self.k * (step - self.x0))))
        elif anneal_function == 'sigmoid':
            return float(1 / (1 + np.exp(0.001 * (self.x0 - step))))
        elif anneal_function == 'negative-sigmoid':
            return float(1 / (1 + np.exp(-0.001 * (self.x0 - step))))
        elif anneal_function == 'linear':
            return min(1, step / self.x0)

    def wd_anneal_function(self, unk_max, anneal_function, step):
        return unk_max * self.kl_anneal_function(anneal_function, step)

    def get_kl_weight(self, step):
        if self.step_kl_weight is None:
            return self.kl_anneal_function(self.args.anneal_function, step)
        else:
            return self.step_kl_weight

    def compute_kl_loss(self, mean, logv, step):
        kl_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        kl_weight = self.get_kl_weight(step)
        return kl_loss, kl_weight

    def score(self, src_sent_tensor, src_sent_lengths, sent_mask, src_parse_tensor, src_parse_lengths, temp_mask,
              tgt_sent_tensor, tgt_parse_tensor, sent_seq_recover, temp_seq_recover, step, sent_type_tensor=None, src_perm_idx=None, 
              tgt_parse_lengths=None, tgt_parse_perm=None):
        # set up sentence and template mask .
        # if self.att is not None:
        #     self.att.set_mask(sent_mask, temp_mask)
        # sentence pattern embedding .
        #print(src_sent_lengths)
        tgt_lengths = src_sent_lengths + 1
        #print(src_parse_lengths)
        tgt_parse_lengths = tgt_parse_lengths - 1
        _, tgt_parse_recover = tgt_parse_lengths.sort(0, False)
        #print(tgt_lengths)
        sent_type_emb = None
        # if sent_type_tensor is not None:
        #     sent_type_emb = self.sent_pattern_emb(sent_type_tensor)
        batch_size = src_sent_tensor.size(0)
        sent_encoder_outputs, sent_encoder_hidden = self.sent_encoder(src_sent_tensor, src_sent_lengths)
        parse_encoder_outputs, parse_encoder_hidden = self.parse_encoder(src_parse_tensor, src_parse_lengths)
        # print(sent_encoder_hidden.size())
        # encoder to decoder bridge.
        # sent_encoder_hidden = self.sent_bridge(sent_encoder_hidden)
        # parse_encoder_hidden = self.parse_bridge(parse_encoder_hidden)
        # need to recover the order .
        #sent_encoder_outputs = sent_encoder_outputs[sent_seq_recover, :, :]
        #sent_encoder_hidden = sent_encoder_hidden[:, sent_seq_recover, :]
        # print(sent_encoder_hidden.size())
        # encoder to decoder bridge.
        #sent_latent, sent_mean, sent_logv = self.sem_hidden_to_latent(sent_encoder_hidden, True)
        # print(sent_latent.size())
        # parse_latent, parse_mean, parse_logv = self.syn_hidden_to_latent(parse_encoder_hidden, True)
        #sent_encoder_hidden = self.latent_to_hidden(self.sem_to_hidden, sent_latent)
        # print(sent_encoder_hidden.size())
        # parse_encoder_hidden = self.latent_to_hidden(self.syn_to_hidden, parse_latent)
        sent_encoder_hidden = self.sent_bridge(sent_encoder_hidden)
        parse_encoder_hidden = self.parse_bridge(parse_encoder_hidden)
        # print(sent_encoder_hidden.size())
        # sent_encoder_hidden = self.sent_tensor_fusion(sent_encoder_hidden, parse_encoder_hidden)
        # interaction between the two decoder .
        
        parse_encoder_outputs = parse_encoder_outputs[temp_seq_recover, :, :]
        parse_encoder_hidden = parse_encoder_hidden[:, temp_seq_recover, :]
        parse_encoder_outputs = parse_encoder_outputs[tgt_parse_perm]
        parse_encoder_hidden = parse_encoder_hidden[:, tgt_parse_perm,:]
        parse_scores, dec_hidden = self.parse_decoder.score(tgt_parse_tensor, parse_encoder_hidden,
                                                            parse_encoder_outputs, tgt_lengths=tgt_parse_lengths)
        # syn dec hidden and sent encoder hidden fusion .
        #print(dec_hidden.size())
        #exit()
        # follow sentence lengths order .
        dec_hidden = dec_hidden[:, tgt_parse_recover,:]
        dec_hidden = dec_hidden[:, src_perm_idx, :]
        sent_encoder_hidden = self.hidden_fusion(dec_hidden, sent_encoder_hidden)
        #print(' fusion', sent_encoder_hidden.size())
        #exit()
        sent_scores, _ = self.sent_decoder.score(inputs=tgt_sent_tensor,
                                                 encoder_hidden=sent_encoder_hidden,
                                                 encoder_outputs=sent_encoder_outputs,
                                                 tgt_lengths=tgt_lengths)

        reconstruct_loss = -(torch.sum(parse_scores) + torch.sum(sent_scores)) / batch_size
        #reconstruct_loss = -torch.sum(sent_scores) / batch_size
        #sem_kl_loss, sem_kl_weight = self.compute_kl_loss(sent_mean, sent_logv, step)
        # syn_kl_loss, syn_kl_weight = self.compute_kl_loss(parse_mean, parse_logv, step)
        # sem_kl_weight *= self.args.kl_factor
        # print(sem_kl_loss.data)
        # print(sem_kl_weight)
        #kl_loss = sem_kl_loss / batch_size
        #kl_loss *= sem_kl_weight
        # syn_kl_weight *= self.args.kl_factor
        # kl_loss = (sem_kl_loss * sem_kl_weight + syn_kl_loss * syn_kl_weight) / batch_size
        # if return_enc_state:
        #     return scores, encoder_hidden
        # else:
        #     return scores
        #
        # pass
        #loss = reconstruct_loss + kl_loss
        loss = reconstruct_loss
        #return loss, sem_kl_weight, kl_loss
        return loss, 1.0, 1.0

    def sent_template_encode(self, src_sent_tensor, src_sent_lengths, src_parse_tensor, src_parse_lengths,
                             sent_seq_recover, temp_seq_recover):
        sent_encoder_outputs, sent_encoder_hidden = self.sent_encoder(src_sent_tensor, src_sent_lengths)
        parse_encoder_outputs, parse_encoder_hidden = self.parse_encoder(src_parse_tensor, src_parse_lengths)

        # encoder to decoder bridge.
        # sent_encoder_hidden = self.sent_bridge(sent_encoder_hidden)
        # parse_encoder_hidden = self.parse_bridge(parse_encoder_hidden)
        # need to recover the order .
        sent_encoder_outputs = sent_encoder_outputs[sent_seq_recover, :, :]
        sent_encoder_hidden = sent_encoder_hidden[:, sent_seq_recover, :]
        parse_encoder_outputs = parse_encoder_outputs[temp_seq_recover, :, :]
        parse_encoder_hidden = parse_encoder_hidden[:, temp_seq_recover, :]
        return sent_encoder_outputs, sent_encoder_hidden, parse_encoder_outputs, parse_encoder_hidden

    def beam_search(self, src_var, src_length, sent_mask, src_parse_tensor, src_parse_length, parse_mask,
                    sent_seq_recover, temp_seq_recover, beam_size=2):
        # if dmts is None:
        #     dmts = self.args.decode_max_time_step
        # src_var = to_input_variable(src_sent, self.src_vocab,
        #                             cuda=self.args.cuda, training=False, append_boundary_sym=False, batch_first=True)
        # src_length = [len(src_sent)]
        # if name == 'sent':
        #     encode = self.sent_encoder
        #     bridger = self.sent_bridge
        #     beam_decoder = self.sent_beam_search
        # elif name == 'parse':
        #     encode = self.parse_encoder
        #     bridger = self.parse_bridge
        #     beam_decoder = self.parse_beam_search
        # else:
        #     print(' Name is not the scope of pre-definition...')
        #     exit(0)
        # set up mask .
        # self.att.set_mask(sent_mask, template_mask)
        sent_encoder_outputs, sent_encoder_hidden = self.sent_encoder(src_var, src_length)
        parse_encoder_outputs, parse_encoder_hidden = self.parse_encoder(src_parse_tensor, src_parse_length)

        # encoder to decoder bridge.
        # sent_encoder_hidden = self.sent_bridge(sent_encoder_hidden)
        # parse_encoder_hidden = self.parse_bridge(parse_encoder_hidden)
        # need to recover the order .
        sent_encoder_outputs = sent_encoder_outputs[sent_seq_recover, :, :]
        sent_encoder_hidden = sent_encoder_hidden[:, sent_seq_recover, :]
        parse_encoder_outputs = parse_encoder_outputs[temp_seq_recover, :, :]
        parse_encoder_hidden = parse_encoder_hidden[:, temp_seq_recover, :]
        # encoder to decoder bridge.
        #sent_latent, sent_mean, sent_logv = self.sem_hidden_to_latent(sent_encoder_hidden)
        # parse_latent, parse_mean, parse_logv = self.syn_hidden_to_latent(parse_encoder_hidden)
        #sent_encoder_hidden = self.latent_to_hidden(self.sem_to_hidden, sent_latent)
        # parse_encoder_hidden = self.latent_to_hidden(self.syn_to_hidden, parse_latent)
        sent_encoder_hidden = self.sent_bridge(sent_encoder_hidden)
        parse_encoder_hidden = self.parse_bridge(parse_encoder_hidden)
        # sent_encoder_output, sent_encoder_hidden, template_encoder_output, template_encoder_hidden = self.sent_template_encode(
        #     src_var, src_length, template_seq_tensor, template_seq_length, seq_recover,
        #     temp_seq_recover)
        # encoder_hidden = self.sent_tensor_fusion(sent_encoder_hidden, template_encoder_hidden)

        # if name == 'sent':
        #     encoder_hidden = bridger.forward(encoder_hidden)
        # else:
        #     encoder_hidden = bridger.forward(input_tensor=encoder_hidden)
        # encoder_hidden =
        parse_meta_data = self.parse_beam_search.beam_search(encoder_hidden=parse_encoder_hidden,
                                                             encoder_outputs=parse_encoder_outputs,
                                                             sent_mask=parse_mask,
                                                             beam_size=beam_size,
                                                             decode_max_time_step=self.args.parse_max_time_step)
        dec_hidden = parse_meta_data['dec_hidden']
        #print(dec_hidden.size())
        sent_encoder_hidden = self.hidden_fusion(dec_hidden, sent_encoder_hidden)
        sent_meta_data = self.sent_beam_search.beam_search(
            encoder_hidden=sent_encoder_hidden,
            encoder_outputs=sent_encoder_outputs,
            sent_mask=sent_mask,
            beam_size=beam_size,
            decode_max_time_step=self.args.sent_max_time_step
        )
        parse_seq, parse_scores = self.recover_sequence(parse_meta_data, self.parse_vocab)
        sent_seq, sent_scores = self.recover_sequence(sent_meta_data, self.word_vocab)

        return parse_seq, parse_scores, sent_seq, sent_scores

    def recover_sequence(self, meta_data, vocab):
        topk_sequence = meta_data['sequence']
        # topk_score = meta_data['score'].squeeze()
        topk_score = meta_data['score']
        # there is a problem.
        # completed_hypotheses = torch.cat(topk_sequence, dim=-1)
        # completed_hypotheses = torch.cat(topk_sequence, dim=0)
        completed_hypotheses = torch.stack(topk_sequence, dim=0)
        # completed_hypotheses = completed_hypotheses[seq_recover]
        number_return = completed_hypotheses.size(0)
        final_result = []
        final_scores = []
        for i in range(number_return):
            hyp = completed_hypotheses[i, :].data.tolist()
            res = dict_id2word(hyp, vocab)
            final_result.append(res)
            final_scores.append(topk_score[i].item())
        return final_result, final_scores

    def save(self, path):
        dir_name = os.path.dirname(path)
        # remove file name, return directory.
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'args': self.args,
            'word_vocab': self.word_vocab,
            'parse_vocab': self.parse_vocab,
            'state_dict': self.state_dict(),
        }
        torch.save(params, path)
