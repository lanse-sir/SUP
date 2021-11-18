import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from autocg.encoder.rnn_encoder import RNNEncoder
from autocg.networks.embeddings import Embeddings
from autocg.utils.nn_funcs import dict_id2word, id2word
import numpy as np
from autocg.Modules.treeencoder import TreeEncoder
from autocg.Modules.tree_gru_1 import TreeGRU
from autocg.Modules.vae_decoder import VAEDecoder
from autocg.Modules.Bag_of_word import BOW

TINY = 1e-9


class model_cvae(nn.Module):
    def __init__(self, args, word_vocab, parse_vocab, bow_vocab=None, word_embedding=None):
        super(model_cvae, self).__init__()
        self.args = args
        self.word_vocab = word_vocab
        self.parse_vocab = parse_vocab
        self.bow_vocab = bow_vocab
        self.dropout_r = args.unk_rate
        # self.mask_ratio = args.mask_ratio
        self.vocab_size = len(self.word_vocab)
        self.emb_size = args.enc_embed_dim
        self.hidden_size = args.hidden_size
        self.sos_id = word_vocab['<s>']
        self.eos_id = word_vocab['</s>']
        self.pad_id = word_vocab['<PAD>']
        self.unk_id = word_vocab['<unk>']

        # vae training ...
        self.latent_size = args.latent_size
        self.k = args.k
        self.x0 = args.x0
        self.step_kl_weight = args.init_step_kl_weight
        self.word_emb = Embeddings(len(self.word_vocab), args.enc_embed_dim, args.enc_ed, add_position_embedding=False,
                                   padding_idx=0)

        if word_embedding is not None:
            self.word_emb.weight.data.copy_(torch.from_numpy(word_embedding))
        self.parse_emb = Embeddings(len(self.parse_vocab), args.tree_embed_dim, args.enc_ed,
                                    add_position_embedding=False, padding_idx=0)

        if self.args.fix_word_emb:
            print("Don't update the weight of word embedding.")
            self.word_emb.weight.requires_grad = False
        # bidirectional ?
        self.enc_factor = 2 if args.bidirectional else 1
        self.enc_hidden_size = args.enc_hidden_dim
        self.enc_dim = args.enc_hidden_dim * self.enc_factor

        if args.mapper_type == "link":
            self.dec_hidden = self.enc_dim
        else:
            self.dec_hidden = args.dec_hidden_dim
        print("Decoder Hidden Size: ", self.dec_hidden)

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

        if self.args.tree_gru:
            self.tree_enc = TreeGRU(args, self.parse_emb, parse_vocab)
            self.sequence_gru = nn.GRU(args.hidden_size, args.hidden_size, num_layers=1, batch_first=True,
                                       bidirectional=False)

        else:
            self.parse_encoder = RNNEncoder(
                vocab_size=len(self.parse_vocab),
                max_len=args.parse_max_time_step,
                input_size=args.enc_embed_dim,
                hidden_size=args.enc_hidden_dim,
                embed_droprate=args.enc_ed,
                rnn_droprate=args.enc_rd,
                n_layers=args.enc_num_layers,
                bidirectional=False,
                rnn_cell=args.rnn_type,
                variable_lengths=True,
                embedding=self.parse_emb
            )

        # use unidirectional gru to obtain the sequence presentation.
        if args.cat_add == 'cat':
            self.sem_syn_map = nn.Linear(self.hidden_size * 3, self.hidden_size)
        elif args.cat_add == 'add':
            self.sent_map = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.sem_syn_map = nn.Linear(self.hidden_size, self.hidden_size)
        else:
            self.sent_map = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.relu = nn.ReLU()
        # VAE part ...
        self.sem_mean = nn.Linear(self.enc_hidden_size, self.latent_size)
        self.sem_logv = nn.Linear(self.enc_hidden_size, self.latent_size)
        # self.sem_to_hidden = nn.Linear(self.latent_dim, self.enc_hidden_size)

        # self.parse_map = nn.Linear(self.enc_hidden_size * self.enc_factor, self.enc_hidden_size)
        # the bag of word loss .
        self.bow = BOW(self.latent_size, self.vocab_size)

        # Decode module
        self.sent_decoder = VAEDecoder(self.vocab_size, self.emb_size, self.word_emb,
                                       self.dec_hidden, self.latent_size, layers=args.enc_num_layers,
                                       dropout=0.0)
        # multi task .
        # self.sent_bridge = MLPBridger(
        #    rnn_type=args.rnn_type,
        #    mapper_type=args.mapper_type,
        #    encoder_dim=self.enc_dim,
        #    encoder_layer=args.enc_num_layers,
        #    decoder_dim=self.dec_hidden,
        #    decoder_layer=args.dec_num_layers
        # )

        print(
            "enc layer: {}, dec layer: {}, type: {}, word dropout: {}, sem kl factor: {}, Sem factor: {}".format(
                args.enc_num_layers,
                args.dec_num_layers,
                args.rnn_type,
                self.dropout_r,
                self.args.kl_sem,
                self.args.mul_sem))

    def to_variable(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def tensor_sort(self, x, recover, src_perm):
        x = x[recover]
        x = x[src_perm]
        return x

    def add_noise(self, variable: Variable, pad_index: int, drop_probability: float = 0.1,
                  shuffle_max_distance: int = 0) -> Variable:
        def perm(i):
            return i[0] + (shuffle_max_distance + 1) * np.random.random()

        new_variable = np.zeros((variable.size(0), variable.size(1)), dtype='int')
        variable = variable.data.cpu().numpy()
        for b in range(variable.shape[0]):
            sequence = variable[b]
            sequence = sequence[sequence != pad_index]
            sequence, reminder = sequence[:-1], sequence[-1:]
            if len(sequence) != 0:
                sequence = sequence[np.random.random_sample(len(sequence)) > drop_probability]
                sequence = [x for _, x in sorted(enumerate(sequence), key=perm)]
            sequence = np.concatenate((sequence, reminder), axis=0)
            sequence = list(np.pad(sequence, (0, variable.shape[1] - len(sequence)), 'constant',
                                   constant_values=pad_index))
        new_variable[b, :] = sequence
        return Variable(torch.LongTensor(new_variable)).cuda()

    def mask_replace(self, sequence):
        if self.mask_ratio > 0.:
            prob = torch.rand(sequence.size())
            if torch.cuda.is_available():
                prob = prob.cuda()
            prob[(sequence.data - self.sos_id) * (sequence.data - self.eos_id) * (
                    sequence.data - self.pad_id) == 0] = 1
            input_sequence = sequence.clone()
            input_sequence[prob < self.mask_ratio] = self.mask_id
            return input_sequence
        return sequence

    def sem_hidden_to_latent(self, hidden, sample=False):
        batch_size = hidden.size(0)
        # hidden_dim = hidden.size(2)
        # hidden = hidden.view(batch_size, hidden_dim * self.enc_factor)
        # hidden = torch.cat((hidden[0], hidden[1]), 1)
        mean = self.sem_mean(hidden)
        logv = self.sem_logv(hidden)
        if sample:
            std = torch.exp(0.5 * logv)
            z = self.to_variable(torch.randn([batch_size, self.latent_size]))
            # have two layer lstm .
            # z = z.repeat(1, 2).view(batch_size, -1, self.latent_dim)
            z = z * std + mean
        else:
            z = mean
        return z, mean, logv

    def syn_hidden_to_latent(self, hidden, sample=False):
        batch_size = hidden.size(1)
        hidden_dim = hidden.size(2)
        # hidden = hidden.view(batch_size, hidden_dim * self.enc_factor)
        hidden = torch.cat((hidden[0], hidden[1]), 1)
        mean = self.syn_mean(hidden)
        logv = self.syn_logv(hidden)
        if sample:
            std = torch.exp(0.5 * logv)
            z = self.to_variable(torch.randn([batch_size, self.latent_dim]))
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
        print('latent to hidden:', hidden.size())
        return hidden

    def hidden_fusion(self, syn_hidden, sent_hidden):
        batch_size = syn_hidden.size(1)
        hidden_cat = torch.cat([sent_hidden, syn_hidden], dim=2).view(-1, self.dec_hidden * 2)
        map_hidden = self.hidden_f(hidden_cat).view(-1, batch_size, self.dec_hidden)
        return map_hidden

    def unk_replace(self, tgt_sequence, unk_ratio=0.0):
        if unk_ratio > 0.:
            # print('tgt sequence:',tgt_sequence)
            prob = torch.rand(tgt_sequence.size())
            if torch.cuda.is_available():
                prob = prob.cuda()
                # prob = prob
            prob[(tgt_sequence.data - self.sos_id) * (tgt_sequence.data - self.eos_id) * (
                    tgt_sequence.data - self.pad_id) == 0] = 1
            dec_input_sequence = tgt_sequence.clone()
            dec_input_sequence[prob < unk_ratio] = self.unk_id
            return dec_input_sequence
        return tgt_sequence

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
        elif anneal_function == 'linear_bound':
            return min(0.8, step / self.x0)

    def wd_anneal_function(self, unk_max, anneal_function, step):
        return unk_max * self.kl_anneal_function(anneal_function, step)

    def get_kl_weight(self, step):
        if self.step_kl_weight is None:
            return self.kl_anneal_function(self.args.anneal_function, step)
        else:
            return self.step_kl_weight

    def get_kl_loss(self, mean, logv):
        kl_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        return kl_loss

    def compute_kl_loss(self, mean, logv, step):
        kl_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        kl_weight = self.get_kl_weight(step)
        return kl_loss, kl_weight

    def list_2_batch(self, list_tensor):
        batch_size = len(list_tensor)
        leave_lengths = torch.LongTensor(list(map(len, list_tensor)))
        max_lengths = leave_lengths.max().item()
        dim = list_tensor[0][0].size(0)
        leave_seq_tensor = torch.zeros(batch_size, max_lengths, dim)
        mask = torch.zeros(batch_size, max_lengths).byte()
        for i in range(batch_size):
            seq_len = leave_lengths[i].item()
            leave_seq_tensor[i, :seq_len] = torch.stack(list_tensor[i])
            mask[i, :seq_len] = torch.Tensor([1] * seq_len)
        leave_lengths, idx_perm = leave_lengths.sort(0, descending=True)
        leave_seq_tensor = leave_seq_tensor[idx_perm]
        _, idx_recover = idx_perm.sort(0, descending=False)
        return self.to_variable(leave_seq_tensor), self.to_variable(leave_lengths), self.to_variable(
            mask), self.to_variable(idx_recover)

    def leave_seq_encode(self, input_tensor, input_lengths=None):
        if input_lengths is not None:
            input_tensor = nn.utils.rnn.pack_padded_sequence(input_tensor, input_lengths, batch_first=True)
        output, hidden = self.sequence_gru(input_tensor)
        if input_lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden

    def tree_gru_encode(self, sent_encoder_outputs, trees):
        # syntactic tree encode.
        sent_encoder_output = torch.max(sent_encoder_outputs, dim=1)[0]
        leaf_embedding_list = self.tree_enc(sent_encoder_output, trees)
        leave_seq_tensor, leave_lengths, leave_mask, idx_recover = self.list_2_batch(leaf_embedding_list)
        # leave node sequence encoding .
        leave_seq_tensor, leave_seq_hidden = self.leave_seq_encode(leave_seq_tensor, leave_lengths)
        leave_seq_tensor = leave_seq_tensor[idx_recover]
        leave_lengths = leave_lengths[idx_recover]
        leave_seq_hidden = leave_seq_hidden[:, idx_recover, :]
        return leave_seq_hidden

    def seq_gru_encode(self, parse_seq_tensor, parse_seq_lengths, parse_seq_recover):
        parse_encoder_outputs, parse_encoder_hidden = self.parse_encoder(parse_seq_tensor, parse_seq_lengths)
        parse_encoder_outputs = parse_encoder_outputs[parse_seq_recover, :, :]
        parse_encoder_hidden = parse_encoder_hidden[:, parse_seq_recover, :]
        return parse_encoder_hidden

    def score(self, input_tensor, step):
        src_sent_tensor = input_tensor['src_seq']
        src_sent_lengths = input_tensor['src_lengths']
        src_sent_recover = input_tensor['src_recover']
        # content_word_seq = input_tensor['con_word']
        tgt_sent_tensor = input_tensor['tgt_sent']
        bag_label = input_tensor['bow']
        # content_word_seq = input_tensor['con_word']
        batch_size = src_sent_tensor.size(0)
        if self.args.noise:
            src_sent_tensor = self.add_noise(src_sent_tensor, self.pad_id)
        # src_sent_tensor = self.mask_replace(src_sent_tensor)
        sent_encoder_outputs, sent_encoder_hidden = self.sent_encoder(src_sent_tensor, src_sent_lengths)
        sent_encoder_outputs = sent_encoder_outputs[src_sent_recover, :, :]
        sent_encoder_hidden = sent_encoder_hidden[:, src_sent_recover, :]
        sent_x = torch.cat((sent_encoder_hidden[0], sent_encoder_hidden[1]), 1)

        # syntactic tree encode.
        if self.args.tree_gru:
            trees = input_tensor['tree']
            parse_encoder_hidden = self.tree_gru_encode(sent_encoder_outputs, trees)
        else:
            parse_seq_tensor = input_tensor['parse_seq']
            parse_seq_lengths = input_tensor['parse_seq_lengths']
            parse_seq_recover = input_tensor['parse_recover']
            parse_encoder_hidden = self.seq_gru_encode(parse_seq_tensor, parse_seq_lengths, parse_seq_recover)
        syn_y = parse_encoder_hidden.squeeze(0)

        if self.args.cat_add == 'cat':
            sent_syn = self.sem_syn_map(torch.cat([sent_x, syn_y], dim=1))
        elif self.args.cat_add == 'add':
            sent_map_x = self.sent_map(sent_x)
            sent_syn = self.sem_syn_map(sent_map_x + syn_y)
        else:
            sent_syn = self.sent_map(sent_x)
        # non-linear translation .
        sent_syn = self.relu(sent_syn)

        sem_z, sem_mean, sem_logv = self.sem_hidden_to_latent(sent_syn, sample=self.training)
        # syn_z, syn_mean, syn_logv = self.syn_hidden_to_latent(parse_encoder_hidden, sample=self.training)
        dynamic_unk = self.wd_anneal_function(self.dropout_r, self.args.unk_schedule, step)

        # Bag of word loss .
        if self.args.bow_factor > 0:
            bow_loss = self.bow(sem_z, bag_label)
        else:
            bow_loss = torch.Tensor([0.]).cuda()
        # next is the decoder section .
        # Decode
        # content_word_emb = self.word_emb(content_word_seq)
        # content_lens = content_word_seq.ne(self.pad_id).sum(dim=-1).float()
        # content_vec = content_word_emb.sum(dim=1)
        # content_vec = content_vec.div((content_lens+1).unsqueeze(1))
        # decoder_hidden = content_vec.unsqueeze(0)

        hidden = None
        initial_output = self.to_variable(self.sent_decoder.initial_output(batch_size))
        input_ids = tgt_sent_tensor[:, :-1]
        input_ids = self.unk_replace(input_ids, dynamic_unk)
        logprods, hidden, _, _ = self.sent_decoder(input_ids, initial_output, sem_z, syn_y, hidden=hidden)

        # compute loss
        # decode loss.
        if tgt_sent_tensor.size(0) == batch_size:
            output_ids = tgt_sent_tensor.contiguous().transpose(1, 0)
        output_ids = output_ids[1:].contiguous().view(-1)
        tgt_sent_log_scores = torch.gather(logprods.view(-1, logprods.size(2)), 1, output_ids.unsqueeze(1)).squeeze(1)
        tgt_sent_log_scores = tgt_sent_log_scores * (1.0 - torch.eq(output_ids, self.pad_id).float())

        # content word section .
        # if content_word_seq.size(0) == batch_size:
        #     content_word_seq = content_word_seq.transpose(1, 0)
        # content_idx = content_word_seq.contiguous().view(-1)
        # content_word_log_scores = torch.gather(logprods.view(-1, logprods.size(2)), 1, content_idx.unsqueeze(1)).squeeze(1)
        # content_word_log_scores = content_word_log_scores * (1.0 - torch.eq(content_idx, self.pad_id).float())
        # content_scores = -torch.sum(content_word_log_scores)/batch_size

        # batch size .
        sent_scores = tgt_sent_log_scores.view(-1, batch_size).sum(dim=0)
        s_reconstruct_loss = -torch.sum(sent_scores) / batch_size
        sem_kl_loss, sem_kl_weight = self.compute_kl_loss(sem_mean, sem_logv, step)
        sem_weight_kl_loss = sem_kl_loss * sem_kl_weight / batch_size

        loss = self.args.mul_sem * s_reconstruct_loss + self.args.kl_sem * sem_weight_kl_loss  + self.args.bow_factor * bow_loss

        ret = {'loss': loss,
               'elbo': s_reconstruct_loss + (sem_kl_loss / batch_size),
               'sem kl loss': sem_kl_loss,
               'bow': bow_loss,
               'sent loss': s_reconstruct_loss,
               'kl_weight': sem_kl_weight
               }
        return ret

    def beam_search(self, input_tensor, beam_size=4, max_ratio=3.2):
        src_var = input_tensor['src_seq']
        src_length = input_tensor['src_lengths']
        src_sent_recover = input_tensor['src_recover']
        # parse_seq_tensor = input_tensor['parse_seq']
        # parse_seq_lengths = input_tensor['parse_seq_lengths']
        # parse_seq_recover = input_tensor['parse_recover']
        batch_size = src_var.size(0)

        # don't need the target sequence .
        sent_encoder_outputs, sent_encoder_hidden = self.sent_encoder(src_var, src_length)
        # need to recover the order .
        sent_encoder_outputs = sent_encoder_outputs[src_sent_recover, :, :]
        sent_encoder_hidden = sent_encoder_hidden[:, src_sent_recover, :]
        sent_x = torch.cat((sent_encoder_hidden[0], sent_encoder_hidden[1]), 1)

        # sem_z, sem_mean, sem_logv = self.sem_hidden_to_latent(sent_encoder_hidden, sample=False)

        # syntactic tree encode .
        if self.args.tree_gru:
            trees = input_tensor['tree']
            parse_encoder_hidden = self.tree_gru_encode(sent_encoder_outputs, trees)
        else:
            parse_seq_tensor = input_tensor['parse_seq']
            parse_seq_lengths = input_tensor['parse_seq_lengths']
            parse_seq_recover = input_tensor['parse_recover']
            parse_encoder_hidden = self.seq_gru_encode(parse_seq_tensor, parse_seq_lengths, parse_seq_recover)
        syn_y = parse_encoder_hidden.squeeze(0)

        if self.args.cat_add == 'cat':
            sent_syn = self.sem_syn_map(torch.cat([sent_x, syn_y], dim=1))

        elif self.args.cat_add == 'add':
            sent_map_x = self.sent_map(sent_x)
            sent_syn = self.sem_syn_map(sent_map_x + syn_y)
        else:
            sent_syn = self.sent_map(sent_x)
        # non-linear translation .
        sent_syn = self.relu(sent_syn)
        sem_z, sem_mean, sem_logv = self.sem_hidden_to_latent(sent_syn, sample=False)

        # start to decode .
        translations = [[] for i in range(batch_size)]
        pending = set(range(batch_size))

        # hidden = hidden.repeat(1, beam_size, 1)
        hidden = None
        # tree_context = leave_seq_tensor.repeat(beam_size, 1, 1)
        # tree_mask = leave_mask.repeat(beam_size, 1)
        syn_z = syn_y.repeat(beam_size, 1)
        sem_z = sem_z.repeat(beam_size, 1)

        # context_lengths *= beam_size
        # context_mask = self.mask(context_lengths)

        ones = beam_size * batch_size * [1]
        prev_words = beam_size * batch_size * [self.sos_id]
        output = self.to_variable(self.sent_decoder.initial_output(beam_size * batch_size))

        translation_scores = batch_size * [-float('inf')]
        hypotheses = batch_size * [(0.0, [])] + (beam_size - 1) * batch_size * [
            (-float('inf'), [])]  # (score, translation)

        while len(pending) > 0:
            # Each iteration should update: prev_words, hidden, output
            input_var = Variable(torch.LongTensor([prev_words]), requires_grad=False).cuda().transpose(0, 1)
            # logprobs, hidden, output = self.decoder(var, ones, self.decoder_embeddings, hidden, context, context_mask,
            #                                         output, self.generator)
            log_softmax_output, hidden, output, _ = self.sent_decoder(input_var, output, sem_z, syn_z, hidden)

            prev_words = log_softmax_output.max(dim=2)[1].squeeze().data.cpu().numpy().tolist()

            word_scores, words = log_softmax_output.topk(k=beam_size + 1, dim=2, sorted=False)
            word_scores = word_scores.squeeze(0).data.cpu().numpy().tolist()  # (beam_size*batch_size) * (beam_size+1)
            words = words.squeeze(0).data.cpu().numpy().tolist()

            for sentence_index in pending.copy():
                candidates = []  # (score, index, word)
                for rank in range(beam_size):
                    index = sentence_index + rank * batch_size
                    for i in range(beam_size + 1):
                        word = words[index][i]
                        score = hypotheses[index][0] + word_scores[index][i]
                        if word != self.eos_id:
                            candidates.append((score, index, word))
                        elif score > translation_scores[sentence_index]:
                            translations[sentence_index] = hypotheses[index][1] + [word]
                            translation_scores[sentence_index] = score
                best = []  # score, word, translation, hidden, output
                for score, current_index, word in sorted(candidates, reverse=True)[:beam_size]:
                    translation = hypotheses[current_index][1] + [word]
                    # best.append(
                    #     (score, word, translation, hidden[:, current_index, :].data, output[current_index].data))
                    best.append(
                        (score, word, translation, hidden[:, current_index, :].data))
                for rank, (score, word, translation, h) in enumerate(best):
                    next_index = sentence_index + rank * batch_size
                    hypotheses[next_index] = (score, translation)
                    prev_words[next_index] = word
                    hidden[:, next_index, :] = h
                    # output[next_index, :] = o
                if len(hypotheses[sentence_index][1]) >= max_ratio * src_length[sentence_index] or \
                        translation_scores[sentence_index] > hypotheses[sentence_index][0]:
                    pending.discard(sentence_index)
                    if len(translations[sentence_index]) == 0:
                        translations[sentence_index] = hypotheses[sentence_index][1]
                        translation_scores[sentence_index] = hypotheses[sentence_index][0]
        res = id2word(translations, self.word_vocab)
        #res = [' '.join(s).replace('@@ ', '') for s in res]
        return res, translation_scores

    def greedy_decode(self, input_tensor, max_ratio=3.2):
        src_var = input_tensor['src_seq']
        src_length = input_tensor['src_lengths']
        src_sent_recover = input_tensor['src_recover']
        # parse_seq_tensor = input_tensor['parse_seq']
        # parse_seq_lengths = input_tensor['parse_seq_lengths']
        # parse_seq_recover = input_tensor['parse_recover']
        content_word_seq = input_tensor['con_word']
        batch_size = src_var.size(0)

        # don't need the target sequence .
        sent_encoder_outputs, sent_encoder_hidden = self.sent_encoder(src_var, src_length)
        # need to recover the order .
        sent_encoder_outputs = sent_encoder_outputs[src_sent_recover, :, :]
        sent_encoder_hidden = sent_encoder_hidden[:, src_sent_recover, :]
        sent_x = torch.cat((sent_encoder_hidden[0], sent_encoder_hidden[1]), 1)
        # sem_z, sem_mean, sem_logv = self.sem_hidden_to_latent(sent_encoder_hidden, sample=False)

        # syntactic tree encode.
        if self.args.tree_gru:
            trees = input_tensor['tree']
            parse_encoder_hidden = self.tree_gru_encode(sent_encoder_outputs, trees)
        else:
            parse_seq_tensor = input_tensor['parse_seq']
            parse_seq_lengths = input_tensor['parse_seq_lengths']
            parse_seq_recover = input_tensor['parse_recover']
            parse_encoder_hidden = self.seq_gru_encode(parse_seq_tensor, parse_seq_lengths, parse_seq_recover)
        syn_y = parse_encoder_hidden.squeeze(0)

        # syn_y = torch.cat((parse_encoder_hidden[0], parse_encoder_hidden[1]), 1)
        if self.args.cat_add == 'cat':
            sent_syn = self.sem_syn_map(torch.cat([sent_x, syn_y], dim=1))

        elif self.args.cat_add == 'add':
            sent_map_x = self.sent_map(sent_x)
            sent_syn = self.sem_syn_map(sent_map_x + syn_y)
        else:
            sent_syn = self.sent_map(sent_x)
        # non-linear translation .
        sent_syn = self.relu(sent_syn)
        sem_z, sem_mean, sem_logv = self.sem_hidden_to_latent(sent_syn, sample=False)

        # content_word_emb = self.word_emb(content_word_seq)
        # content_lens = content_word_seq.ne(self.pad_id).sum(dim=-1).float()
        # content_vec = content_word_emb.sum(dim=1)
        # content_vec = content_vec.div((content_lens+1).unsqueeze(1))
        # hidden = content_vec.unsqueeze(0)
        hidden = None
        output = self.to_variable(self.sent_decoder.initial_output(batch_size))
        translations = [[] for i in range(batch_size)]
        pending = set(range(batch_size))
        prev_words = batch_size * [self.sos_id]

        # attentions = [[] for i in range(batch_size)]
        # logits = [[] for i in range(batch_size)]
        while len(pending) > 0:
            # Each iteration should update: prev_words, hidden, output
            input_var = Variable(torch.LongTensor([prev_words]), requires_grad=False).cuda().transpose(0, 1)
            # logprobs, hidden, output = self.decoder(var, ones, self.decoder_embeddings, hidden, context, context_mask,
            #                                         output, self.generator)
            log_softmax_output, hidden, output, _ = self.sent_decoder(input_var, output, sem_z, syn_y, hidden=hidden)
            prev_words = log_softmax_output.max(dim=2)[1].squeeze().data.cpu().numpy().tolist()
            # logit = logit.squeeze(0)
            # att_list = att.squeeze(0).data.cpu().numpy().tolist()
            for i in pending.copy():
                if prev_words[i] == self.eos_id:
                    pending.discard(i)
                else:
                    translations[i].append(prev_words[i])
                    # attentions[i].append(att_list[i])
                    # logits[i].append(logit[i])
                    if len(translations[i]) >= max_ratio * src_length[i]:
                        pending.discard(i)
        res = id2word(translations, self.word_vocab)
        
        return res, 0.0

    def differentiable_greedy_decode(self, input_tensor, sample=True, gumbel_softmax=True, beta=1.0, max_ratio=3.2):
        src_var = input_tensor['src_seq']
        src_length = input_tensor['src_lengths']
        src_sent_recover = input_tensor['src_recover']
        # parse_seq_tensor = input_tensor['parse_seq']
        # parse_seq_lengths = input_tensor['parse_seq_lengths']
        # parse_seq_recover = input_tensor['parse_recover']
        batch_size = src_var.size(0)

        # don't need the target sequence .
        sent_encoder_outputs, sent_encoder_hidden = self.sent_encoder(src_var, src_length)
        # need to recover the order .
        sent_encoder_outputs = sent_encoder_outputs[src_sent_recover, :, :]
        sent_encoder_hidden = sent_encoder_hidden[:, src_sent_recover, :]
        sent_x = torch.cat((sent_encoder_hidden[0], sent_encoder_hidden[1]), 1)
        # sem_z, sem_mean, sem_logv = self.sem_hidden_to_latent(sent_encoder_hidden, sample=False)

        # syntactic tree encode.
        if self.args.tree_gru:
            trees = input_tensor['tree']
            parse_encoder_hidden = self.tree_gru_encode(sent_encoder_outputs, trees)
        else:
            parse_seq_tensor = input_tensor['parse_seq']
            parse_seq_lengths = input_tensor['parse_seq_lengths']
            parse_seq_recover = input_tensor['parse_recover']
            parse_encoder_hidden = self.seq_gru_encode(parse_seq_tensor, parse_seq_lengths, parse_seq_recover)
        syn_y = parse_encoder_hidden.squeeze(0)
        # syn_y = torch.cat((parse_encoder_hidden[0], parse_encoder_hidden[1]), 1)
        if self.args.cat_add == 'cat':
            sent_syn = self.sem_syn_map(torch.cat([sent_x, syn_y], dim=1))

        elif self.args.cat_add == 'add':
            sent_map_x = self.sent_map(sent_x)
            sent_syn = self.sem_syn_map(sent_map_x + syn_y)
        else:
            sent_syn = self.sent_map(sent_x)
        # non-linear translation .
        sent_syn = self.relu(sent_syn)
        sem_z, sem_mean, sem_logv = self.sem_hidden_to_latent(sent_syn, sample=sample)
        # compute kl loss.
        kl_loss = self.get_kl_loss(sem_mean, sem_logv)

        hidden = None
        output = self.to_variable(self.sent_decoder.initial_output(batch_size))
        # translations = [[] for i in range(batch_size)]
        pending = set(range(batch_size))
        prev_words = batch_size * [self.sos_id]
        input_var = torch.LongTensor([prev_words]).cuda().transpose(0, 1)
        # attentions = [[] for i in range(batch_size)]
        soft_words = []
        for i in range(self.args.sent_max_time_step):
            # Each iteration should update: prev_words, hidden, output
            # logprobs, hidden, output = self.decoder(var, ones, self.decoder_embeddings, hidden, context, context_mask,
            #                                         output, self.generator)
            log_softmax_output, hidden, output, logit = self.sent_decoder(input_var, output, sem_z, syn_y, hidden)
            # prev_words = log_softmax_output.max(dim=2)[1].squeeze().data.cpu().numpy().tolist()
            if gumbel_softmax:
                input_var = self.gumbel_softmax(logit, beta=beta).transpose(0, 1)
            else:
                input_var = log_softmax_output.exp().transpose(0, 1)
            # prev_words = input_var.max(dim=2)[1].squeeze().data.cpu().numpy().tolist()
            soft_words.append(input_var)
            # att_list = att.squeeze(0).data.cpu().numpy().tolist()
            #for i in pending.copy():
                #if prev_words[i] == self.eos_id:
                #    pending.discard(i)
                #else:
                #    translations[i].append(prev_words[i])
                    # attentions[i].append(att_list[i])
                    # logits[i].append(logit[i])
                    #if len(translations[i]) >= max_ratio * src_length[i]:
                        #pending.discard(i)
        #res = id2word(translations, self.word_vocab)
        soft_words = torch.cat(soft_words, dim=1)
        return soft_words, kl_loss

    def gumbel_softmax(self, inputs, beta=0.5, tau=1.0):
        noise = inputs.data.new(*inputs.size()).uniform_()
        noise.add_(TINY).log_().neg_().add_(TINY).log_().neg_()
        return F.softmax((inputs + beta * Variable(noise)) / tau, dim=-1)

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

    def bow_predict_seq(self, idx, vocab):
        id2word = {idx: word for word, idx in vocab.items()}
        # print(idx.size())
        batch_size = idx.size(0)
        topk = idx.size(1)
        res = [[] for i in range(batch_size)]
        for x in range(batch_size):
            for y in range(topk):
                # print(idx[x,y])
                res[x].append(id2word[idx[x, y].item()])
        return res

    def save(self, path):
        dir_name = os.path.dirname(path)
        # remove file name, return directory.
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'args': self.args,
            'word_vocab': self.word_vocab,
            'parse_vocab': self.parse_vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)
