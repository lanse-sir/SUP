import argparse
import random
import os
import sys

sys.path.append('../../../')
sys.path.append('../../')
# random.seed(0)
import numpy as np
import torch
import codecs
from cvae_model.classifer.model_seq2seq import seq2seq
from autocg.subwordnmt.apply_bpe import BPE, read_vocabulary
from cvae_model.data_init import data_to_idx
from autocg.utils.config_funcs import yaml_to_dict, dict_to_args
from autocg.optimizers.schedule import LinearWarmupRsqrtDecay
from autocg.pretrain_embedding import load_embedding
from autocg.load_file import save_to_file, load_file_sent_or_parser
from autocg.evaluation_utils import run_multi_bleu
from visdom import Visdom

# evaluation bleu script .
MULTI_BLEU_PERL = '/data/users/yeg/eval_tools/multi-bleu.perl'


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training Syntactic Text Generation Models",
        usage="train.py [<args>] [-h | --help]"
    )
    # input files
    parser.add_argument('--run_name', type=str, default='run_test')
    parser.add_argument('--model_config', type=str, help='models configs')
    # parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--dev_every', type=int, default=500)
    parser.add_argument('--start_eval', type=int, default=-1)
    # parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--reset_optimizer', type=bool, default=False)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--reload', type=str)
    parser.add_argument('--sent', type=str)
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
        lr = optimizer.param_groups[0]['lr'] * main_args.lr_decay
        print('decay learning rate to %f' % lr, file=sys.stdout)
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
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # reset patience
        patience = 0
    # lr = optimizer.param_groups[0]['lr']
    # if lr <= 1e-6:
    # print('early stop!', file=sys.stdout)
    return model, optimizer, patience


def batch_serialization(batch, src_vocab, parse_vocab, if_train=True):
    batch_size = len(batch)
    sents = [pair[0] for pair in batch]
    sents_idx_seq = [data_to_idx(s, src_vocab) for s in sents]

    # as for src sentence .
    def padding(sents):
        word_seq_lengths = torch.LongTensor(list(map(len, sents)))
        max_seq_len = word_seq_lengths.max().item()
        word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
        # tgt_word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
        mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).byte()
        for idx, (seq, seqlen) in enumerate(zip(sents, word_seq_lengths)):
            seqlen = seqlen.item()
            word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
            mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
        return word_seq_lengths, word_seq_tensor, mask

    src_seq_lengths, src_seq_tensor, src_mask = padding(sents_idx_seq)
    # from big to small sort.
    src_seq_lengths, src_perm_idx = src_seq_lengths.sort(0, descending=True)
    src_seq_tensor = src_seq_tensor[src_perm_idx]
    _, src_sent_seq_recover = src_perm_idx.sort(0, descending=False)
    if if_train:
        tgt_sents = [['<s>'] + pair[1] + ['</s>'] for pair in batch]
        tgt_idx_seq = [data_to_idx(s, parse_vocab) for s in tgt_sents]
        tgt_seq_lengths, tgt_seq_tensor, tgt_sent_mask = padding(tgt_idx_seq)
    else:
        tgt_seq_tensor = torch.Tensor([0.0])

    if torch.cuda.is_available():
        return {
            'src_lengths': src_seq_lengths.cuda(),
            'src_seq': src_seq_tensor.cuda(),
            'src_mask': src_mask.cuda(),
            'tgt_seq': tgt_seq_tensor.cuda(),
            'src_recover': src_sent_seq_recover.cuda()
        }
    else:
        return {
            'src_lengths': src_seq_lengths,
            'src_seq': src_seq_tensor,
            'src_mask': src_mask,
            'tgt_seq': tgt_seq_tensor,
            'src_recover': src_sent_seq_recover
        }


def evaluate(model, test_data, word_vocab, parse_vocab, opt):
    preds = []
    refs = []
    batch_size = opt.eval_bs
    model.eval()
    if opt.greedy:
        print('Using greedy decode ......')
    else:
        print('Using beam search decode ......')
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            input_tensors = batch_serialization(batch, word_vocab, parse_vocab, if_train=False)
            # print(src_parse_seq_lengths)
            if opt.greedy:
                sent_results, sent_scores = model.greedy_decode(input_tensors)
                # att_ws += atts
            else:
                sent_results, sent_scores = model.beam_search(input_tensors, beam_size=opt.beam_size)
            preds += sent_results
            refs += [pair[1] for pair in batch]
    # if len(att_ws) != 0:
    #     np.save('generate/1_test.npy', att_ws)
    filename = opt.sent
    print('Save generate sentence to {} .'.format(filename))
    save_to_file(preds, filename)
    acc = syntactic_acc(preds, refs)
    return acc


def syntactic_acc(preds, refs):
    total_num = len(refs)
    right = 0
    for pred, ref in zip(preds, refs):
        # string then split.
        if type(ref) is str:
            ref = ref.split()

        str_pred, str_ref = ' '.join(pred), ' '.join(ref)
        if str_pred == str_ref:
            right += 1
    return right / total_num


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
    bpe_codes = codecs.open(opt.bpe_codes_path, encoding='utf-8')
    bpe_vocab = codecs.open(opt.bpe_vocab_path, encoding='utf-8')
    bpe_vocab = read_vocabulary(bpe_vocab, opt.bpe_vocab_thresh)
    bpe = BPE(bpe_codes, '@@', bpe_vocab, None)
    
    train_sent = load_file_sent_or_parser(opt.input_sent, 'sent')
    train_ref = load_file_sent_or_parser(opt.input_ref, 'sent')
    
    # load development set.
    valid_sent = load_file_sent_or_parser(opt.dev_sent, 'parse')
    valid_sent = [bpe.segment(sent).split() for sent in valid_sent]
    
    valid_ref = load_file_sent_or_parser(opt.dev_ref, 'sent')
    
    
    train_data = list(zip(train_sent, train_ref))
    valid_data = list(zip(valid_sent, valid_ref))
    print('All train instance is %d, dev instance is %d' % (len(train_data), len(valid_data)))
    print(train_data[0][0])
    print(train_data[0][1])
    print('Validition Set.')
    print(valid_data[0][0])
    if opt.reload is not None:
        print("Loading Pretrained Model ......")
        params = torch.load(opt.reload, map_location=lambda storage, loc: storage)
        opt = params['args']
        vocab, parse_vocab = params['word_vocab'], params['parse_vocab']
        pretrained_model = params['state_dict']
    else:
        vocab = load_vocab(opt.vocab)
        parse_vocab = load_vocab(opt.parser_vocab)
        # bow_vocab = load_vocab(opt.bow_vocab)
    print("Vocab Size :", len(vocab))
    # set up use GPU id
    torch.cuda.set_device(opt.gpu)
    # load pretrained word embedding .
    if os.path.exists(opt.pretrain_emb) and opt.debug is not True and pretrained_model is None:
        word_embedding = load_embedding(opt.pretrain_emb, vocab)
    else:
        word_embedding = None

    model = seq2seq(opt, vocab, parse_vocab, word_embedding=word_embedding)
    if pretrained_model is not None:
        model.load_state_dict(pretrained_model)
    if opt.cuda and torch.cuda.is_available():
        model = model.cuda()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # hyper parameter .
    batch_size = opt.batch_size

    # start to train .
    step = 0
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
            seq_tensor = batch_serialization(batch, vocab, parse_vocab, if_train=True)
            ret = model(seq_tensor, step)
            # adv training start .
            loss = ret['loss']
            total_loss += loss.item() * batch_size
            optimizer.zero_grad()
            loss.backward()
            if opt.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
            optimizer.step()
            # record the loss of training stage .
            # record the loss per step with visdom.
            # vis.line([total_loss / train_num], [step], win='train loss', update='append', opts=dict(title='Loss'))
            # vis.line([total_sent_loss / train_num], [step - 1], win='sent loss', update='append',
            #          opts=dict(title='Sentence Loss'))
            # vis.line([att_loss], [step], win='attn loss', update='append', opts=dict(title='Attention Loss'))
            lr = optimizer.param_groups[0]['lr']
            if step % opt.log_every == 0:
                print(
                    '[Epoch %d] [Iter %d] Train LOSS=%.3f, lr=%.5f' % (i, step, total_loss / train_num, lr))
            if i > opt.start_eval and step % opt.dev_every == 0:
                acc = evaluate(model=model,
                               test_data=valid_data,
                               word_vocab=vocab,
                               parse_vocab=parse_vocab,
                               opt=opt)
                print('[Epoch %d] [Iter %d] Syntactic predict accuracy is %.3f' % (i, step, acc))
                is_better = (history_scores == []) or acc > max(history_scores)
                if is_better:
                    # model_file_idx += 1
                    history_scores.append(acc)
                # model_file = opt.model_file + "_" + str(model_file_idx)
                model_file = opt.model_file
                m, optimizer, patience = get_lr_schedule(is_better, model, optimizer, opt, patience, model_file)
                m.train()


if __name__ == "__main__":
    config_args = parse_args()
    # args = config_args['opt']
    main(config_args)
