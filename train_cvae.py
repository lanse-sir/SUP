import argparse
import random
import os
import sys
import pickle
import codecs

sys.path.append('../../')
sys.path.append('../')
# random.seed(0)
import numpy as np
import torch
from model_cvae import model_cvae
from autocg.subwordnmt.apply_bpe import BPE, read_vocabulary
from data_init import batch_serialization_parallel, batch_sentence_tree
from autocg.utils.config_funcs import yaml_to_dict, dict_to_args
from autocg.optimizers.schedule import LinearWarmupRsqrtDecay
from autocg.pretrain_embedding import load_embedding
from autocg.load_file import save_to_file, load_file_sent_or_parser
from autocg.evaluation_utils import run_multi_bleu
from visdom import Visdom

# evaluation bleu script .
MULTI_BLEU_PERL =  '/dfsdata2/yangeg1_data/multi-bleu.perl'


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training Syntactic Text Generation Models",
        usage="train.py [<args>] [-h | --help]"
    )
    # input files
    parser.add_argument('--run_name', type=str, default='run_test')
    parser.add_argument('--pretrain_emb', type=str, default='../data/glove.42B.300d.txt')
    parser.add_argument('--model_config', type=str, help='models configs',
                        default='cvae_para50w.yaml')
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--dev_every', type=int, default=1000)
    parser.add_argument('--start_eval', type=int, default=0)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--reset_optimizer', type=bool, default=False)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--reload', type=str, default='')
    parser.add_argument('--sent', type=str)
    # parser.add_argument('--parse', type=str, default='predict_result/parse_preds.txt')
    parser.add_argument('--beam_size', type=int, default=4)
    opt = parser.parse_args()
    main_args, model_args = None, None

    if opt.model_config is not None:
        model_args = dict_to_args(yaml_to_dict(opt.model_config)['model_configs'])

    return {
        'base': main_args,
        'model': model_args,
        "opt": opt
    }


def load_vocab(fname):
    vocab = {}
    with open(fname, 'r', encoding='utf-8') as f_r:
        for line in f_r:
            word, idx = line.strip().split()
            vocab[word] = int(idx)
    return vocab


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

def generate_sentence(model, test_data, word_vocab, parse_vocab, word_occurence, document_num, opt):
    sent_preds = []
    batch_size = opt.eval_bs
    model.eval()
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
                sent_results, _ = model.greedy_decode(input_tensors)
            else:
                sent_results, _ = model.beam_search(input_tensors, beam_size=opt.beam_size)
            sent_preds += sent_results
    return sent_preds


def evaluate(model, test_data, word_vocab, parse_vocab, word_occurence, document_num, opt):
    sent_preds = []
    batch_size = opt.eval_bs
    model.eval()
    if opt.greedy:
        print('Using greedy decode ......')
    else:
        print('Using beam search decode ......')
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            input_tensors = batch_serialization_parallel(batch, word_vocab, parse_vocab,opt,
                                                         word_occurence=word_occurence,
                                                         document_num=document_num,
                                                         if_train=False,
                                                         mode='decode')
            if opt.greedy:
                sent_results, sent_scores = model.greedy_decode(input_tensors)
                # att_ws += atts
            else:
                sent_results, sent_scores = model.beam_search(input_tensors, beam_size=opt.beam_size)
            sent_preds += sent_results
    filename = opt.sent
    print('Save generate sentence to {} .'.format(filename))
    # remove @@
    res = [' '.join(s).replace('@@ ', '') for s in sent_preds]
    save_to_file(res, filename)
    
    ori_bleu = -1.0
    ref_bleu = -1.0
    if opt.dev_sent is not None:
        # parse_bleu = run_multi_bleu(parse_file, opt.dev_parser, MULTI_BLEU_PERL)
        ori_bleu = run_multi_bleu(filename, opt.dev_sent, MULTI_BLEU_PERL)
    if opt.dev_ref is not None:
        ref_bleu = run_multi_bleu(filename, opt.dev_ref, MULTI_BLEU_PERL)
    return ori_bleu, ref_bleu


def adv_training(d_p, ret, args, opt):
    # sent_dis = model.adv_sentence_decoder.parameters()
    # for name, _ in model.adv_sentence_decoder.named_parameters():
    #     print(name)
    # parse_dis = model.adv_parse_decoder.parameters()
    optimizer = torch.optim.RMSprop(d_p, lr=opt.adv_lr)
    # parse_optimizer = torch.optim.RMSprop(parse_dis, lr=0.001)
    if args.sem_adv:
        adv_sent_loss = ret['adv sent loss']
        optimizer.zero_grad()
        adv_sent_loss.backward(retain_graph=True)
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(d_p, args.clip)
        optimizer.step()
    if args.syn_adv:
        adv_parse_loss = ret['adv parse loss']
        optimizer.zero_grad()
        adv_parse_loss.backward(retain_graph=True)
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(d_p, args.clip)
        optimizer.step()


def merge_dict(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def dict_print(d):
    for key, value in d.items():
        print('{0:25}'.format(key), '==> ', value)


def main(config_args):
    pretrained_model = None
    opt = config_args['opt']
    model_args = config_args['model']
    # merge the base args and model args.
    args_dict = merge_dict(vars(opt), vars(model_args))
    opt = argparse.Namespace(**args_dict)

    # set up initial seed .
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # vis = Visdom(env=opt.run_name)
    dict_print(args_dict)
    print('-' * 120)
    
    # load bpe codes
    if opt.bpe:
        bpe_codes = codecs.open(opt.bpe_codes_path, encoding='utf-8')
        bpe_vocab = codecs.open(opt.bpe_vocab_path, encoding='utf-8')
        bpe_vocab = read_vocabulary(bpe_vocab, opt.bpe_vocab_thresh)
        bpe = BPE(bpe_codes, '@@', bpe_vocab, None)
    
    
    # load input file.
    train_sent = load_file_sent_or_parser(opt.input_sent, 'sent')
    train_ref = load_file_sent_or_parser(opt.input_ref, 'sent')
    
    if opt.tree_gru:
        train_parser = load_file_sent_or_parser(opt.input_parser, 'parse')
        valid_parser = load_file_sent_or_parser(opt.dev_parser, 'parse')
    else:
        train_parser = load_file_sent_or_parser(opt.input_parser, 'sent')
        valid_parser = load_file_sent_or_parser(opt.dev_parser, 'sent')
    
    if opt.bpe:
        valid_sent = load_file_sent_or_parser(opt.dev_sent, 'parse') # using bpe.
        valid_sent = [bpe.segment(sent).split() for sent in valid_sent]
    else:
        valid_sent = load_file_sent_or_parser(opt.dev_sent, 'sent')
    
    
    
    train_data = list(zip(train_sent, train_parser, train_ref))
    valid_data = list(zip(valid_sent, valid_parser))
    print('All train instance is %d, dev instance is %d' % (len(train_data), len(valid_data)))
    print('Train data ......')
    print(train_data[0][0])
    print(train_data[0][1])
    print(train_data[0][2])
    
    print('Dev data ......')
    print(valid_data[0][0])
    
    if opt.reload != '':
        print("Loading Pretrained Model ......")
        params = torch.load(opt.reload, map_location=lambda storage, loc: storage)
        vocab, parse_vocab = params['word_vocab'], params['parse_vocab']
        pretrained_model = params['state_dict']
    else:
        vocab = load_vocab(opt.vocab)
        parse_vocab = load_vocab(opt.parser_vocab)
        # bow_vocab = load_vocab(opt.bow_vocab)
    print("Vocab Size :", len(vocab))
    # load word occurence vocab .
    # if opt.bow:
    word_occurence, document_num = pickle.load(open(opt.word_occur, 'rb'))

    # set up use GPU id
    torch.cuda.set_device(opt.gpu)
    # load pretrained word embedding .
    if os.path.exists(opt.pretrain_emb) and opt.debug is not True and pretrained_model is None:
        word_embedding = load_embedding(opt.pretrain_emb, vocab)
    else:
        word_embedding = None

    m = model_cvae(opt, vocab, parse_vocab, word_embedding=word_embedding)
    if pretrained_model is not None:
        m.load_state_dict(pretrained_model)
    if opt.cuda and torch.cuda.is_available():
        m = m.cuda()
    print(m)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), lr=opt.lr)

    # hyper parameter .
    batch_size = opt.batch_size

    # start to train .
    step = 0
    sentence_loss = 0.
    train_num = 0
    total_loss = 0.0
    total_sent_loss = 0.0
    history_scores = []
    patience = 0
    model_file_idx = 0
    for i in range(opt.epoch):
        random.shuffle(train_data)
        for idx in range(0, len(train_data), batch_size):
            # set training mode .
            batch = train_data[idx:idx + batch_size]
            # set up learning rate .
            step += 1
            train_num += len(batch)
            if opt.linear_warmup:
                lr = LinearWarmupRsqrtDecay(warm_up=4000, init_lr=0.0, max_lr=opt.lr, step=step)
                optimizer.param_groups[0]['lr'] = lr
            # ... generate batch tensor .
            seq_tensor = batch_serialization_parallel(batch, vocab, parse_vocab,
                                                      word_occurence=word_occurence,
                                                      document_num=document_num,
                                                      opt = opt,
                                                      if_train=True,
                                                      mode='train')
            ret = m.score(seq_tensor, step)
            # adv training start .
            loss, sent_loss, bow = ret['loss'], ret['sent loss'], ret['bow']
            kl_weight = ret['kl_weight']
            total_sent_loss += sent_loss.item() * batch_size
            total_loss += loss.item() * batch_size
            sem_kl_loss = ret['sem kl loss'].item()
            bow_loss = bow.item()
            optimizer.zero_grad()
            loss.backward()
            if opt.clip > 0:
                torch.nn.utils.clip_grad_norm_(m.parameters(), opt.clip)
            optimizer.step()
            # record the loss of training stage .
            # record the loss per step with visdom.
            # vis.line([total_loss / train_num], [step], win='train loss', update='append', opts=dict(title='Loss'))
            # vis.line([total_sent_loss / train_num], [step - 1], win='sent loss', update='append',
            #          opts=dict(title='Sentence Loss'))
            # vis.line([att_loss], [step], win='attn loss', update='append', opts=dict(title='Attention Loss'))
            if step % opt.log_every == 0:
                print(
                    '[Epoch %d] [Iter %d] Train LOSS=%.3f, sentence loss=%.3f, sem kl=%.3f, bag of word loss=%.3f, kl weight=%.4f' % (
                        i, step, total_loss / train_num, total_sent_loss/train_num, sem_kl_loss / batch_size, bow_loss, kl_weight))
            if i > opt.start_eval and step % opt.dev_every == 0:
                ori_bleu, ref_bleu = evaluate(model=m,
                                              test_data=valid_data,
                                              word_vocab=vocab,
                                              parse_vocab=parse_vocab,
                                              word_occurence=word_occurence,
                                              document_num=document_num,
                                              opt=opt)
                print('[Epoch %d] [Iter %d] Dev Ori BLEU=%.3f, Ref BLEU=%.3f' % (i, step, ori_bleu, ref_bleu))
                if ref_bleu == -1:
                    ref_bleu = ori_bleu
                is_better = (history_scores == []) or ref_bleu > max(history_scores)
                if is_better:
                    model_file_idx += 1
                    history_scores.append(ref_bleu)
                # model_file = opt.model_file + "_" + str(model_file_idx)
                model_file = opt.model_file
                m, optimizer, patience = get_lr_schedule(is_better, m, optimizer, opt, patience, model_file)
                m.train()


if __name__ == "__main__":
    config_args = parse_args()
    # args = config_args['opt']
    main(config_args)
