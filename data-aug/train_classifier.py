import sys
sys.path.append('../../')
sys.path.append('../')

import torch
import os
from autocg.pretrain_embedding import load_embedding
from classifer import binary_classifer
import random
import argparse



#fix the random seed .
#seed = 0
#random.seed(seed)
#torch.manual_seed(seed)


def load_vocab(fname):
    vocab = {}
    with open(fname, 'r', encoding='utf-8') as f_r:
        for line in f_r:
            word, idx = line.strip().split()
            vocab[word] = int(idx)
    return vocab


def load_sentence(file_name):
    sentences = []
    with open(file_name, 'r', encoding='utf-8') as f_r:
        for line in f_r:
            sentences.append(line.strip().split())
    return sentences


def load_sentiment_label(file_name):
    sentences = []
    with open(file_name, 'r', encoding='utf-8') as f_r:
        for line in f_r:
            sentences.append(int(line.strip()))
    return sentences


def data_to_idx(sent, vocab):
    return [vocab.get(w, vocab['<unk>']) for w in sent]


def batch_serialization(batch, src_vocab, if_train=True):
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
    # tgt_parse_seq_lengths, tgt_parse_seq_tensor, tgt_parse_mask = padding(tgt_parsers_idx_seq)
    # from big to small sort.
    src_seq_lengths, src_perm_idx = src_seq_lengths.sort(0, descending=True)
    src_seq_tensor = src_seq_tensor[src_perm_idx]
    src_mask = src_mask[src_perm_idx]
    # sentence type :
    _, sent_seq_recover = src_perm_idx.sort(0, descending=False)
    target = [pair[1] for pair in batch]
    target_tensor = torch.LongTensor(target)[src_perm_idx]
    return src_seq_lengths, src_seq_tensor, src_mask, sent_seq_recover, target_tensor
    
    #return src_seq_lengths, src_seq_tensor, src_mask, sent_seq_recover


def evaluate(model, dev_data, vocab, batch_size=10):
    model.eval()
    right_instance = 0
    with torch.no_grad():
        for start in range(0, len(dev_data), batch_size):
            batch = dev_data[start:start + batch_size]
            src_seq_lengths, src_seq_tensor, src_mask, sent_seq_recover, target_tensor = list(
                map(lambda x: x.cuda(), batch_serialization(batch,
                                                            vocab,
                                                            if_train=False)))
            inputs = {'src tensor': src_seq_tensor, 'src lengths': src_seq_lengths, ' src mask': src_mask,
                      'seq recover': sent_seq_recover, 'target': target_tensor}
            target_tensor = target_tensor[sent_seq_recover]
            logits = model.predict(inputs)
            score, idx = logits.topk(1)
            right_instance += torch.sum(idx.squeeze(1) == target_tensor).item()
    return right_instance / len(dev_data)


def main(args):
    print(args)
    print('-'*150)
    # load train, dev, test data .
    train_sent = load_sentence(args.train_sent)
    train_label = load_sentiment_label(args.train_label)
    dev_sent = load_sentence(args.dev_sent)
    dev_label = load_sentiment_label(args.dev_label)
    test_sent = load_sentence(args.test_sent)
    test_label = load_sentiment_label(args.test_label)
    train_data = list(zip(train_sent, train_label))
    dev_data = list(zip(dev_sent, dev_label))
    test_data = list(zip(test_sent, test_label))
    print('All train instance is %d, dev instance is %d, test instance is %d' % (
        len(train_data), len(dev_data), len(test_data)))
    vocab = load_vocab(args.vocab)
    # set up use GPU id
    torch.cuda.set_device(0)
    # load pretrained word embedding .
    if os.path.exists(args.pretrain_emb):
        word_embedding = load_embedding(args.pretrain_emb, vocab)
    else:
        word_embedding = None
    model = binary_classifer(args, vocab, word_embedding)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # training set up
    batch_size = args.batch_size
    step = 0
    patience = 0
    total_loss = 0.0
    history_acc = []
    for e in range(args.epoch):
        random.shuffle(train_data)
        for start in range(0, len(train_data), batch_size):
            batch = train_data[start:start + batch_size]
            step += 1
            src_seq_lengths, src_seq_tensor, src_mask, sent_seq_recover, target_tensor = list(
                map(lambda x: x.cuda(), batch_serialization(batch,
                                                            vocab,
                                                            if_train=True)))
            # build a dict to store these tensor .
            inputs = {'src tensor': src_seq_tensor, 'src lengths': src_seq_lengths, ' src mask': src_mask,
                      'seq recover': sent_seq_recover, 'target': target_tensor}
            loss = model(inputs)
            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            if step % args.log_every == 0:
                print("[Epoch %d] [Step %d] Loss: %.6f" % (e, step, total_loss / step))
            if e > args.start_eval and step % args.dev_every == 0:
                dev_acc = evaluate(model, dev_data, vocab)
                test_acc = evaluate(model, test_data, vocab)
                print("[Epoch %d] [Step %d] DEV Acc is %.4f, TEST Acc is %.4f ." % (e, step, dev_acc, test_acc))
                is_better = (history_acc == []) or dev_acc > max(history_acc)
                if is_better:
                    history_acc.append(dev_acc)
                model, optimizer, patience = get_lr_schedule(is_better, model, optimizer, args, patience)
                model.train()
    print('Training Finished !!! ')


def get_lr_schedule(is_better, model, optimizer, main_args, patience, reload_model=True):
    model_file = main_args.model_file
    lr = main_args.lr
    if is_better:
        patience = 0
        print('save model to [%s]' % model_file, file=sys.stdout)
        model.save(model_file)
        # also save the optimizers' state
        #torch.save(optimizer.state_dict(), model_file + '.optim.bin')
    elif patience < main_args.patience:
        patience += 1
        print('hit patience %d' % patience, file=sys.stdout)

    if patience == main_args.patience:
        # num_trial += 1
        # print('hit #%d trial' % num_trial, file=sys.stdout)
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
        #if main_args.reset_optimizer:
            #print('reset optimizer', file=sys.stdout)
            #optimizer = torch.optim.Adam(model.inference_model.parameters(), lr=lr)
        #else:
            #print('restore parameters of the optimizers', file=sys.stdout)
            #optimizer.load_state_dict(torch.load(model_file + '.optim.bin'))

        # set new lr
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr

        # reset patience
        patience = 0
    # lr = optimizer.param_groups[0]['lr']
    # if lr <= 1e-6:
    # print('early stop!', file=sys.stdout)
    return model, optimizer, patience


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training Syntactic Text Generation Models",
        usage="train.py [<args>] [-h | --help]"
    )
    # input files ../data/glove.42B.300d.txt
    parser.add_argument("--train_sent", type=str, default='data/remain_train_sent.txt',
                        help="Path of train sentence .")
    parser.add_argument('--train_label', type=str, default='data/remain_train_label.txt')
    parser.add_argument("--dev_sent", type=str, default="data/remain_dev_sent.txt")
    parser.add_argument("--dev_label", type=str, default="data/remain_dev_label.txt")
    parser.add_argument("--test_sent", type=str, default="data/remain_test_sent.txt")
    parser.add_argument("--test_label", type=str, default="data/remain_test_label.txt")

    parser.add_argument("--vocab", type=str, default='vocabulary/word_vocab',
                        help="Path of source and target vocabulary")
    parser.add_argument('--pretrain_emb', type=str, default='../data/glove.42B.300d.txt')
    parser.add_argument('--enc_embed_dim', type=int, default=300)
    parser.add_argument('--label_num', type=int, default=2)
    parser.add_argument("--rnn_type", type=str, default='lstm')
    parser.add_argument("--src_max_time_step",type=int, default=100)
    parser.add_argument('--enc_hidden_dim', type=int, default=300)
    parser.add_argument('--enc_ed', type=float, default=0.5)
    parser.add_argument('--enc_rd', type=float, default=0.0)
    parser.add_argument('--enc_num_layers', type=int, default=1)
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--warm_up', type=int, default=4000)
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--load_from', type=str, default=None)
    parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--dev_every', type=int, default=200)
    parser.add_argument('--start_eval', type=int, default=-1)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--model_file', default='checkpoint/classifer_model')
    parser.add_argument('--reset_optimizer', type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
