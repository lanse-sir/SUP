# define some auxiliary loss .
import sys
import codecs
sys.path.append('../')
sys.path.append('../../')
import argparse
import random
import pickle
import torch
import numpy as np
import torch.nn.functional as F
from autocg.utils.config_funcs import yaml_to_dict, dict_to_args
from autocg.subwordnmt.apply_bpe import BPE, read_vocabulary
from autocg.load_file import load_file_sent_or_parser, save_to_file
from cvae_model.data_init import pack_batch, batch_serialization_parallel, batch_process, bow_onehot, data_to_idx
from cvae_model.train_cvae import merge_dict, dict_print
from cvae_model.integrate import INTE_Model
from cvae_model.train_cvae import MULTI_BLEU_PERL
from autocg.evaluation_utils import run_multi_bleu
from cvae_model.classifer.trainer import batch_serialization, syntactic_acc
from cvae_model.data_init import trees_to_dict
TINY = 1e-9
# templates = ["( ROOT ( S ( NP ) ( VP ) ( . ) ) )",
#             "( ROOT ( S ( VP ) ( . ) ) )",
#             "( ROOT ( S ( CC ) ( NP ) ( VP ) ( . ) ) )",
#             "( ROOT ( S ( NP ) ( ADVP ) ( VP ) ( . ) ) )",
#             "( ROOT ( S ( ADVP ) ( NP ) ( VP ) ( . ) ) )",
#             "( ROOT ( S ( S ) ( , ) ( NP ) ( VP ) ( . ) ) )",
#             "( ROOT ( S ( S ) ( , ) ( CC ) ( S ) ( . ) ) )",
#             "( ROOT ( S ( PP ) ( , ) ( NP ) ( VP ) ( . ) ) )",
#             "( ROOT ( S ( ADVP ) ( , ) ( NP ) ( VP ) ( . ) ) )",
#             "( ROOT ( S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )"]


def get_lr_schedule(is_better, model, optimizer, main_args, patience, model_file, reload_model=True):
    lr = main_args.lr
    if is_better:
        patience = 0
        print('save model to [%s]' % model_file, file=sys.stdout)
        model.save(model_file)
        # also save the optimizers' state
        torch.save(optimizer.state_dict(), model_file + '.optim.bin')
    elif patience < main_args.patience:
        patience += 1
        print('hit patience %d' % patience, file=sys.stdout)

    if patience == main_args.patience:
        print(' hit the max patience number .')
        # decay lr, and restore from previously best checkpoint
        # lr = optimizer.param_groups[0]['lr'] * main_args.lr_decay
        # print('decay learning rate to %f' % lr, file=sys.stdout)
        # load model
        if reload_model:
            print('load previously best model', file=sys.stdout)
            params = torch.load(model_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(params['state_dict'])
            if torch.cuda.is_available():
                model = model.cuda()

        # load optimizers
        if main_args.reset_optimizer:
            print('reset optimizer', file=sys.stdout)
            optimizer = torch.optim.Adam(model.inference_model.parameters(), lr=lr)
        else:
            print('restore parameters of the optimizers', file=sys.stdout)
            optimizer.load_state_dict(torch.load(model_file + '.optim.bin'))

        # set new lr
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        # reset patience
        patience = 0
    # lr = optimizer.param_groups[0]['lr']
    # if lr <= 1e-6:
    # print('early stop!', file=sys.stdout)
    return model, optimizer, patience


def get_lengths(tokens, eos_idx):
    lengths = torch.cumsum(tokens == eos_idx, 1)
    lengths = (lengths == 0).long().sum(-1)
    lengths = lengths  # don't need the <eos> token
    return lengths


def load_template(path):
    templates = []
    with open(path, 'r', encoding='utf-8') as f_r:
        for line in f_r:
            template, _ = line.strip().split('----')
            templates.append(template.strip())
    return templates


def evaluate(model_g, model_d, test_data, word_vocab, parse_vocab, word_occurence, document_num, opt):
    sent_preds = []
    tgt_syntaxs = [pair[1] for pair in test_data]
    syn_preds = []
    batch_size = opt.eval_bs

    model_g.eval()

    if opt.greedy:
        print('Using greedy decode ......')
    else:
        print('Using beam search decode ......')
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            input_tensors = batch_serialization_parallel(batch, word_vocab, parse_vocab, opt,
                                                         word_occurence=word_occurence,
                                                         document_num=document_num,
                                                         if_train=False)
            if opt.greedy:
                sent_results, _ = model_g.greedy_decode(input_tensors)
                # att_ws += atts
            else:
                sent_results, _ = model_g.beam_search(input_tensors, beam_size=opt.beam_size)
            sent_preds += sent_results
    
    filename = opt.sent
    print('Generate sentence to {} .'.format(filename))
    
    recover_bpe = [' '.join(s).replace('@@ ', '') for s in sent_preds]
    save_to_file(recover_bpe, filename)
    
    ori_bleu = -1.0
    ref_bleu = -1.0
    syn_acc = -1.0
    if opt.dev_sent is not None:
        # parse_bleu = run_multi_bleu(parse_file, opt.dev_parser, MULTI_BLEU_PERL)
        ori_bleu = run_multi_bleu(filename, opt.dev_sent, MULTI_BLEU_PERL)
    if opt.dev_ref is not None:
        ref_bleu = run_multi_bleu(filename, opt.dev_ref, MULTI_BLEU_PERL)
    print(sent_preds[0])
    # syntactic accuracy .
    if not opt.no_acc:  # compute syntax accuracy .
        assert len(tgt_syntaxs) == len(sent_preds)
        syn_input_datas = list(zip(sent_preds, tgt_syntaxs))
        for i in range(0, len(syn_input_datas), batch_size):
            batch = syn_input_datas[i:i + batch_size]
            syn_input_tensors = batch_serialization(batch, word_vocab, parse_vocab, if_train=False)
            sent_results, sent_scores = model_d.greedy_decode(syn_input_tensors)
            syn_preds += sent_results
        syn_file = opt.syn_pred
        syn_acc = syntactic_acc(syn_preds, tgt_syntaxs)
        save_to_file(syn_preds, syn_file)
    return ori_bleu, ref_bleu, syn_acc


def main(args):
    # set up initial seed .
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # load bpe codes
    bpe_codes = codecs.open(opt.bpe_codes_path, encoding='utf-8')
    bpe_vocab = codecs.open(opt.bpe_vocab_path, encoding='utf-8')
    bpe_vocab = read_vocabulary(bpe_vocab, opt.bpe_vocab_thresh)
    bpe = BPE(bpe_codes, '@@', bpe_vocab, None)
    
    src_sents = load_file_sent_or_parser(args.input_sent, 'sent')
    src_template = load_file_sent_or_parser(args.input_parser, 'parse')
    assert len(src_sents) == len(src_template)
    
    templates = load_template(args.template)
    if args.tree_gru:
        input_parse = [template for template in templates]
    else:
        input_parse = [template.split() for template in templates]

    dev_sents = load_file_sent_or_parser(args.dev_sent, 'parse')
    dev_sents = [bpe.segment(sent).split() for sent in dev_sents]
    
    dev_parses = load_file_sent_or_parser(args.dev_parser, 'parse')
    
    dev_datas = list(zip(dev_sents, dev_parses))
    print('-' * 120)
    print('Input Sentence size is {}, dev data size is {}'.format(len(src_sents), len(dev_parses)))
    print('Train data ......')
    print(src_sents[0])
    print(src_template[0])
    
    print('Dev data ......')
    print(dev_datas[0][0])
    
    word_occurence, document_num = pickle.load(open(opt.word_occur, 'rb'))
    torch.cuda.set_device(args.gpu)

    # cvae_params = torch.load(args.cvae_file)
    # cvae = model_cvae(cvae_params['args'], cvae_params['word_vocab'], cvae_params['parse_vocab'])
    # cvae.load_state_dict(cvae_params['state_dict'])

    # params = torch.load(args.style_transfer_file, map_location=lambda storage, loc: storage)
    # classifer = seq2seq(params['args'], params['word_vocab'], params['parse_vocab'])
    # for p in classifer.parameters():
    #     p.requires_grad=False

    # classifer.load_state_dict(params['state_dict'])
    # assert len(cvae_params['word_vocab']) == len(params['word_vocab'])
    # assert len(cvae_params['parse_vocab']) == len(params['parse_vocab'])

    model = INTE_Model(args, funetine_config=True)

    # for p in classifer.parameters():
    #     total_parameters.append(p)

    # for p in cvae.parameters():
    #     total_parameters.append(p)
    word_vocab = model.cvae.word_vocab
    parse_vocab = model.cvae.parse_vocab

    tune_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(tune_params, lr=args.lr)
    if torch.cuda.is_available():
        model = model.cuda()
    # cvae_params = model.cvae_params

    batch_size = args.batch_size
    step = 0

    self_sentence = 0.
    self_kl = 0.
    self_total_loss = 0.

    total_cycle_loss = 0.
    cycle_step = 1
    cycle_instance_num = 0
    cycle_sentence_loss = 0.
    
    cls_loss = 0.
    
    total_word_match_loss = 0.
    instance_num = 0
    history_scores = []
    patience = 0
    train_minibatches = [(start, start + batch_size) for start in range(0, len(src_sents), batch_size)]
    for e in range(args.epoch):
        random.shuffle(train_minibatches)
        for (start, end) in train_minibatches:
            #print(start, ' ', end)
            step += 1
            batch_sents = src_sents[start:end]
            # the templates of input sentences.
            batch_templates = src_template[start:end]
            
            instance_num += len(batch_sents)
            optimizer.zero_grad()
            # self-reconstruction.
            if args.use_self:
                self_batch = list(zip(batch_sents, batch_templates, batch_sents))
                self_input_tensor = batch_serialization_parallel(self_batch, word_vocab,
                                                                 parse_vocab,
                                                                 opt=args,
                                                                 word_occurence=word_occurence,
                                                                 document_num=document_num,
                                                                 if_train=True,
                                                                 mode='train')
                self_ret = model.cvae.score(self_input_tensor, step)
                self_loss, self_sent_loss, self_bow, self_kl_loss = self_ret['loss'], self_ret['sent loss'], self_ret[
                    'bow'], self_ret['sem kl loss']
                self_sentence += self_sent_loss.item()
                self_kl += self_kl_loss.item() / batch_size
                self_total_loss += self_loss.item()
                # self-reconstruction loss.
                self_loss.backward()

            # cycle reconstruction.
            if args.cls_factor > 0.0 and step > args.cls_step:
                cycle_batch = pack_batch(batch_sents, input_parse)
                input_tensor = batch_serialization_parallel(cycle_batch, word_vocab, parse_vocab,
                                                            opt=args,
                                                            word_occurence=word_occurence,
                                                            document_num=document_num,
                                                            if_train=True,
                                                            mode='decode')
                gene_probs, _ = model.cvae.differentiable_greedy_decode(input_tensor=input_tensor, sample=args.sample, gumbel_softmax=args.gumbel)
                
                # empty_ids = []
                # for i in range(batch_size):
                #     if len(logits[i]) == 0:
                #         empty_ids.append(i)
                # seq_probs_tensor, seq_lengths_tensor, _, seq_recover_tensor = list_2_batch(batch_probs, if_train=True)

                # soft input.
                # outputs_idx_seq = [data_to_idx(s, word_vocab) for s in outputs]
                # outputs_lengths, _, outputs_mask = batch_process(outputs_idx_seq, if_train=True)
                gene_lengths = get_lengths(gene_probs.argmax(-1), word_vocab['</s>'])
                outputs_lengths, outputs_perm = gene_lengths.sort(0, descending=True)
                seq_probs_tensor = gene_probs[outputs_perm]
                _, seq_recover_tensor = outputs_perm.sort(0, descending=False)

                # cycle learning section.
                # target sequence batch process.
                tgt_sents = [['<s>'] + sent + ['</s>'] for sent in batch_sents]
                tgt_idx_seq = [data_to_idx(s, word_vocab) for s in tgt_sents]
                tgt_lengths_tensor, tgt_seq_tensor, _ = batch_process(tgt_idx_seq, if_train=True)
                # bag of word one hot
                bow_label, content_word_ids = bow_onehot(batch_sents, word_vocab, word_occurence,
                                                         document_num, args)
                content_seq_lengths, content_seq_tensor, _ = batch_process(content_word_ids, if_train=True)
                content_seq_tensor = content_seq_tensor.cuda()
                content_seq_lengths = content_seq_lengths.cuda()
                # parse tree batch process .
                if args.tree_gru:
                    tgt_trees = trees_to_dict(batch_templates)
                    cycle_inputs = {'src_lengths': outputs_lengths.cuda(),
                                    'src_seq': seq_probs_tensor.cuda(),
                                    'src_mask': outputs_perm.cuda(),
                                    'src_recover': seq_recover_tensor.cuda(),
                                    'tree': tgt_trees,
                                    'tgt_sent': tgt_seq_tensor.cuda(),
                                    'bow': bow_label.cuda()
                                    }

                else:
                    temp_idx_seq = [data_to_idx(s, parse_vocab) for s in batch_templates]
                    temp_lengths_tensor, temp_seq_tensor, temp_mask = batch_process(temp_idx_seq, if_train=True)
                    temp_lengths_tensor, temp_perm = temp_lengths_tensor.sort(0, descending=True)
                    temp_seq_tensor = temp_seq_tensor[temp_perm]
                    _, temp_recover_perm = temp_perm.sort(0, descending=False)
                    # pack cycle input.
                    cycle_inputs = {'src_lengths': outputs_lengths.cuda(),
                                    'src_seq': seq_probs_tensor.cuda(),
                                    'src_mask': outputs_perm.cuda(),
                                    'src_recover': seq_recover_tensor.cuda(),
                                    'parse_seq_lengths': temp_lengths_tensor.cuda(),
                                    'parse_seq': temp_seq_tensor.cuda(),
                                    'parse_mask': temp_mask.cuda(),
                                    'parse_recover': temp_recover_perm.cuda(),
                                    'tgt_sent': tgt_seq_tensor.cuda(),
                                    'bow': bow_label.cuda()
                                    }

                # mask empty sequence .
                # empty_mask = torch.ones(batch_size).byte()
                # empty_mask.scatter_(-1, torch.LongTensor(empty_ids), 0)
                # class_label_tensor = input_tensor['c_tgt_parse'][empty_mask]

                # classifer section.
                # classifer label.
                
                class_label_tensor = input_tensor['c_tgt_parse']
                try:
                    classifer_loss = style_transfer_loss(model.classifer, seq_probs_tensor, outputs_lengths,
                                                         seq_recover_tensor,
                                                         class_label_tensor)
                    cycle_loss_dict = cycle_reconstruct_loss(model.cvae, cycle_inputs, step)
                    #word_match_loss = greedy_matching(model.sim_word_emb, logits, content_seq_tensor, content_seq_lengths)
                except RuntimeError:
                    print('Error, generate empty sequence !!!')
                    # self-training continous.
                    if args.clip > 0:
                        torch.nn.utils.clip_grad_norm_(tune_params, args.clip)
                    optimizer.step()
                    continue
                
                
                
                # total loss .
                cycle_instance_num += batch_size
                cycle_step += 1
                cycle_loss, cycle_kl_loss, cycle_res = cycle_loss_dict['loss'], cycle_loss_dict['sem kl loss'], \
                                                       cycle_loss_dict['sent loss']
                cycle_sentence_loss += cycle_res.item()
                cls_loss += classifer_loss.item()
                total_cycle_loss += cycle_loss.item()

                #total_word_match_loss += word_match_loss.item()
                #(args.cls_factor * classifer_loss + args.cycle_factor * cycle_loss + args.wm_factor * word_match_loss).backward()
                (args.cls_factor * classifer_loss + args.cycle_factor * cycle_loss).backward()
            
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(tune_params, args.clip)
            
            optimizer.step()
            if e > args.start_eval and step % args.log_every == 0:
                print(
                    '[Epoch %d] [Iter %d] Self LOSS=%.3f, Cycle Loss=%.3f, Self-Res loss=%.3f, Classifer loss=%.3f, Cycle-Res loss=%.3f, KL loss=%.3f, Word Match=%.3f' % (
                        e, step, self_total_loss / step, total_cycle_loss / cycle_step, self_sentence / step,
                        cls_loss / cycle_step, cycle_sentence_loss / cycle_step, self_kl / step, total_word_match_loss/cycle_step))
            if e > args.start_eval and step % args.dev_every == 0:
                ori_bleu, ref_bleu, syn_acc = evaluate(model_g=model.cvae,
                                                       model_d=model.classifer,
                                                       test_data=dev_datas,
                                                       word_vocab=word_vocab,
                                                       parse_vocab=parse_vocab,
                                                       word_occurence=word_occurence,
                                                       document_num=document_num,
                                                       opt=args)
                
                if ref_bleu == -1:
                    ref_bleu = ori_bleu
                
                over_metric = ref_bleu * args.ref_factor + syn_acc * 100
                print('[Epoch %d] [Iter %d] Syntax Accuracy=%.3f, Dev Ori BLEU=%.3f, Ref BLEU=%.3f, Overall Metrics=%.3f' % (
                    e, step, syn_acc, ori_bleu, ref_bleu, over_metric))
                
                if args.metric == 'overall':
                    score = over_metric
                elif args.metric == 'ref-bleu':
                    score = ref_bleu
                elif args.metric == 'ori-bleu':
                    score = ori_bleu
                else:
                    score = syn_acc
                
                #is_better = (history_scores == []) or score > max(history_scores)
                #if is_better:
                    #history_scores.append(score)
                model_file = opt.model_file + "_" + str(step)
                
                print('save model to [%s]' % model_file, file=sys.stdout)
                model.cvae.save(model_file)
                # model_file = args.model_file
                #model.cvae, optimizer, patience = get_lr_schedule(is_better, model.cvae, optimizer, opt, patience,
                                                                  #model_file)
                model.cvae.train()


def greedy_matching(embedding_layer, soft_word, ref_word, ref_seqs_lengths):
    #print(ref_seqs_lengths)
    ref_embs = embedding_layer(ref_word)
    soft_word_emb = embedding_layer(soft_word)
    # soft_word_emb = soft_word_emb.data.masked_fill_(, 0.)
    ref_embs_norm = ref_embs.norm(dim=-1, keepdim=True)
    soft_word_emb_norm = soft_word_emb.norm(dim=-1, keepdim=True)
    cos_matrix = torch.bmm(ref_embs, soft_word_emb.transpose(1, 2))
    normalization = torch.bmm(ref_embs_norm, soft_word_emb_norm.transpose(1, 2))
    # smooth to avoid divide zero.
    cos_matrix = cos_matrix / (normalization+TINY)
    gm = cos_matrix.max(-1)[0].sum(dim=1)/(ref_seqs_lengths+1)
    gm = gm.mean()
    # .sum() / ref_seqs_lengths
    content_preservation_loss = 1 - gm
    # ref_sent_v = ref_embs.sum(dim=1)
    # soft_sent_v = soft_word_emb.sum(dim=1)
    # content_preservation_loss = F.mse_loss(soft_sent_v, ref_sent_v)
    return content_preservation_loss


def cycle_reconstruct_loss(model_G, inputs, step):
    loss_dict = model_G.score(inputs, step)
    # cycle_loss = loss_dict['loss']
    return loss_dict


def style_transfer_loss(model_D, seq_probs_tensor, seq_lengths_tensor, seq_recover_tensor, label):
    # word_emb = model.word_emb.weight
    # batch_size = seq_probs_tensor.size(0)
    # input_dim = word_emb.size(1)
    # src_tensor = seq_probs_tensor.view(-1, seq_probs_tensor.size(2)).mm(word_emb).view(batch_size, -1, input_dim)
    # with torch.no_grad():
    # model.eval()
    # with torch.no_grad():
    input_tensor = {'src_seq': seq_probs_tensor, 'src_lengths': seq_lengths_tensor.cuda(),
                    'src_recover': seq_recover_tensor.cuda(),
                    'tgt_seq': label}
    # with torch.no_grad():
    loss_dict = model_D(input_tensor)
    adv_loss = loss_dict['loss']
    return adv_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fine Tune CVAE Generation Models",
        usage="fine_tune.py [<args>] [-h | --help]")
    parser.add_argument('--config', type=str, default='fine_tune_config.yaml')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--start_eval', type=int, default=-1)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--dev_every', type=int, default=1000)
    parser.add_argument('--reset_optimizer', type=bool, default=False)
    parser.add_argument('--cvae_file', type=str, default='checkpoint/model_56')
    
    # output file.
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--sent', type=str, default='finetune/predict/fine_tune_dev_sent')
    parser.add_argument('--syn_pred', type=str, default='finetune/predict/fine_tune_syn_pred')
    parser.add_argument('--model_file', type=str, default='finetune/checkpoint/finetune_cvae')
    args = parser.parse_args()
    model_args = dict_to_args(yaml_to_dict(args.config)['model_configs'])

    args_dict = merge_dict(vars(args), vars(model_args))
    opt = argparse.Namespace(**args_dict)
    dict_print(args_dict)
    
    opt.sent = opt.sent + '_' + opt.run_name
    opt.syn_pred = opt.syn_pred + '_' + opt.run_name
    opt.model_file = opt.model_file + '_' + opt.run_name
    main(opt)
