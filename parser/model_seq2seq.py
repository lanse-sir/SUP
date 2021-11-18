import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from autocg.encoder.rnn_encoder import RNNEncoder
from autocg.networks.embeddings import Embeddings
from autocg.networks.bridger import MLPBridger
from autocg.utils.nn_funcs import dict_id2word, id2word
from autocg.Modules.base_decoder import BASEDecoder


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # sum .
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss


class seq2seq(nn.Module):
    def __init__(self, args, word_vocab, parse_vocab, bow_vocab=None, word_embedding=None):
        super(seq2seq, self).__init__()
        self.args = args
        self.word_vocab = word_vocab
        self.parse_vocab = parse_vocab
        self.bow_vocab = bow_vocab
        # self.mask_ratio = args.mask_ratio
        self.vocab_size = len(self.word_vocab)
        self.emb_size = args.enc_embed_dim

        self.sos_id = word_vocab['<s>']
        self.eos_id = word_vocab['</s>']
        self.pad_id = word_vocab['<PAD>']
        self.unk_id = word_vocab['<unk>']

        self.word_emb = Embeddings(len(self.word_vocab), args.embed_size, args.enc_ed, add_position_embedding=False,
                                   padding_idx=0)
        if word_embedding is not None:
            self.word_emb.weight.data.copy_(torch.from_numpy(word_embedding))
        self.parse_emb = Embeddings(len(self.parse_vocab), args.tree_embed_size, args.enc_ed,
                                    add_position_embedding=False, padding_idx=0)
        # bidirectional ?
        self.enc_factor = 2 if args.bidirectional else 1
        self.enc_hidden_size = args.enc_hidden_dim
        self.enc_dim = args.enc_hidden_dim * self.enc_factor

        if args.mapper_type == "mapping":
            self.dec_hidden = self.enc_dim
        else:
            self.dec_hidden = args.dec_hidden_dim

        print("Decoder Hidden Size: ", self.dec_hidden)
        # self.bow_model = bow_model(self.enc_dim, len(self.word_vocab))
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

        self.sent_bridge = MLPBridger(
            rnn_type=args.rnn_type,
            mapper_type=args.mapper_type,
            encoder_dim=self.enc_dim,
            encoder_layer=args.enc_num_layers,
            decoder_dim=self.dec_hidden,
            decoder_layer=args.dec_num_layers
        )

        self.decoder = BASEDecoder(vocab_size=len(parse_vocab), emb_size=args.tree_embed_size, embedding=self.parse_emb,
                                   hidden_size=self.dec_hidden, use_last_output=False)
        print(
            "enc layer: {}, dec layer: {}, type: {}".format(
                args.enc_num_layers,
                args.dec_num_layers,
                args.rnn_type))

    def to_variable(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def forward(self, input_tensor, step=None):
        src_sent_tensor = input_tensor['src_seq']
        src_sent_lengths = input_tensor['src_lengths']
        src_sent_recover = input_tensor['src_recover']

        tgt_sent_tensor = input_tensor['tgt_seq']

        batch_size = src_sent_tensor.size(0)
        # if self.args.noise:
        #     src_sent_tensor = self.add_noise(src_sent_tensor, self.pad_id)
        # src_sent_tensor = self.mask_replace(src_sent_tensor)
        sent_encoder_outputs, sent_encoder_hidden = self.sent_encoder(src_sent_tensor, src_sent_lengths)
        sent_encoder_outputs = sent_encoder_outputs[src_sent_recover, :, :]
        sent_encoder_hidden = sent_encoder_hidden[:, src_sent_recover, :]

        dec_hidden = self.sent_bridge(sent_encoder_hidden)
        # next is the decoder section .
        # Decode
        initial_output = self.to_variable(self.decoder.initial_output(batch_size))
        input_ids = tgt_sent_tensor[:, :-1]
        logprods, hidden, _ = self.decoder(input_ids, initial_output, dec_hidden)

        
        logprods = logprods.transpose(0,1).contiguous().view(-1, logprods.size(2))
        sents_loss = cal_loss(logprods, tgt_sent_tensor[:, 1:], self.pad_id, smoothing=self.args.smoothing)
        s_reconstruct_loss = sents_loss / batch_size
        
        # compute loss
        # decode loss.
        #if tgt_sent_tensor.size(0) == batch_size:
        #    output_ids = tgt_sent_tensor.contiguous().transpose(1, 0)
        #output_ids = output_ids[1:].contiguous().view(-1)
        #tgt_sent_log_scores = torch.gather(logprods.view(-1, logprods.size(2)), 1, output_ids.unsqueeze(1)).squeeze(1)
        #tgt_sent_log_scores = tgt_sent_log_scores * (1.0 - torch.eq(output_ids, self.pad_id).float())

        # batch size .
        #sent_scores = tgt_sent_log_scores.view(-1, batch_size).sum(dim=0)
        #s_reconstruct_loss = -torch.sum(sent_scores) / batch_size

        loss = s_reconstruct_loss

        ret = {'loss': loss}
        return ret

    def beam_search(self, input_tensor, beam_size=4, max_ratio=3.2):
        src_var = input_tensor['src_seq']
        src_length = input_tensor['src_lengths']
        src_sent_recover = input_tensor['src_recover']
        batch_size = src_var.size(0)

        # don't need the target sequence .
        sent_encoder_outputs, sent_encoder_hidden = self.sent_encoder(src_var, src_length)
        # need to recover the order .
        sent_encoder_outputs = sent_encoder_outputs[src_sent_recover, :, :]
        sent_encoder_hidden = sent_encoder_hidden[:, src_sent_recover, :]
        dec_hidden = self.sent_bridge(sent_encoder_hidden)
        # start to decode .
        translations = [[] for i in range(batch_size)]
        pending = set(range(batch_size))

        hidden = dec_hidden.repeat(1, beam_size, 1)
        # hidden = None
        # context_lengths *= beam_size
        # context_mask = self.mask(context_lengths)

        ones = beam_size * batch_size * [1]
        prev_words = beam_size * batch_size * [self.sos_id]
        output = self.to_variable(self.decoder.initial_output(beam_size * batch_size))

        translation_scores = batch_size * [-float('inf')]
        hypotheses = batch_size * [(0.0, [])] + (beam_size - 1) * batch_size * [
            (-float('inf'), [])]  # (score, translation)

        while len(pending) > 0:
            # Each iteration should update: prev_words, hidden, output
            input_var = Variable(torch.LongTensor([prev_words]), requires_grad=False).cuda().transpose(0, 1)
            # logprobs, hidden, output = self.decoder(var, ones, self.decoder_embeddings, hidden, context, context_mask,
            #                                         output, self.generator)
            log_softmax_output, hidden, output = self.decoder(input_var, output, hidden)

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
        res = id2word(translations, self.parse_vocab)
        return res, translation_scores

    def greedy_decode(self, input_tensor, max_ratio=3.2):
        src_var = input_tensor['src_seq']
        src_length = input_tensor['src_lengths']
        src_sent_recover = input_tensor['src_recover']
        batch_size = src_var.size(0)
        # src_length = None
        # don't need the target sequence .
        sent_encoder_outputs, sent_encoder_hidden = self.sent_encoder(src_var, src_length)
        # need to recover the order .
        sent_encoder_outputs = sent_encoder_outputs[src_sent_recover, :, :]
        sent_encoder_hidden = sent_encoder_hidden[:, src_sent_recover, :]
        dec_hidden = self.sent_bridge(sent_encoder_hidden)

        hidden = dec_hidden
        output = self.to_variable(self.decoder.initial_output(batch_size))
        translations = [[] for i in range(batch_size)]
        pending = set(range(batch_size))
        prev_words = batch_size * [self.sos_id]

        # attentions = [[] for i in range(batch_size)]
        while len(pending) > 0:
            # Each iteration should update: prev_words, hidden, output
            input_var = Variable(torch.LongTensor([prev_words]), requires_grad=False).cuda().transpose(0, 1)
            # logprobs, hidden, output = self.decoder(var, ones, self.decoder_embeddings, hidden, context, context_mask,
            #                                         output, self.generator)
            log_softmax_output, hidden, output = self.decoder(input_var, output, hidden)
            prev_words = log_softmax_output.max(dim=2)[1].squeeze().data.cpu().numpy().tolist()
            # att_list = att.squeeze(0).data.cpu().numpy().tolist()
            for i in pending.copy():
                if prev_words[i] == self.eos_id:
                    pending.discard(i)
                else:
                    translations[i].append(prev_words[i])
                    # attentions[i].append(att_list[i])
                    if len(translations[i]) >= max_ratio * src_length[i]:
                        pending.discard(i)
        res = id2word(translations, self.parse_vocab)

        return res, 0.0

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
