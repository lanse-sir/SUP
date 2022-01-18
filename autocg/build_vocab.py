import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import collections
import argparse
from autocg.load_file import load_data_pair
from nltk.corpus import stopwords
import os

stoplist = stopwords.words('english')
punct = ['.', ',', "'s", "''", "``", ':', "'", "`", '?', '-rrb-', '-lrb', '-', '$', '!', ';', '%', '/', '-rsb-',
         '-lsb-', '&', '--', '#', '@', '£', '-lrb-']


def count_word(fname, lower):
    counter = collections.Counter()
    with open(fname, 'r', encoding='utf-8') as f_r:
        for line in f_r:
            if lower:
                word_list = line.strip().lower().split()
            else:
                word_list = line.strip().split()
            counter.update(word_list)

    count_pairs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    words, count = list(zip(*count_pairs))
    return words, count


def count_word_from_list(list_or_tuple):
    counter = collections.Counter()
    for src, template, tgt in list_or_tuple:
        counter.update(src)
        counter.update(template)

    count_pairs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    words, count = list(zip(*count_pairs))
    return words, count


def vocab_save(filename, vocab):
    # np.save(filename, vocab)
    with open(filename, 'w', encoding='utf-8') as f_w:
        for w, idx in vocab.items():
            f_w.write(w + '\t' + str(idx) + '\n')


def main(args):
    # init vocab
    # corpus_fname = args.corpus
    limit = args.limit
    count = 0
    # vocab = {'<PAD>': 0, '<s>': 1, '</s>': 2, '<unk>': 3, '<mask>': 4, '[slot]': 5}

    vocab = {'<PAD>': 0, '<s>': 1, '</s>': 2, '<unk>': 3, '<CLS>':4}

    if os.path.isfile(args.corpus):
        words, counts = count_word(args.corpus, args.lower)
    else:

        print("No such file.")
        # corpus = load_data_pair(args.corpus, name='train')
        # words, counts = count_word_from_list(corpus)

    for word, freq in zip(words, counts):
        if limit and len(vocab) >= limit:
            break

        if word in vocab:
            print("Warning: found duplicate token %s, ignored" % word)
            continue
        if freq > args.cut_freq:
            if args.stopword:
                if word in stoplist or word in punct:
                    continue
            vocab[word] = len(vocab)
            count += freq

    print("Total words: %d" % sum(counts))
    print("Unique words: %d" % len(words))
    print("Remain words: %d " % len(vocab))
    print("Vocabulary coverage: %4.2f%%" % (100.0 * count / sum(counts)))
    vocab_save(args.output, vocab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-corpus', type=str, default='../datas/paranmt-350k/train/train_tgt_parse.txt')
    parser.add_argument('-limit', type=int, default=500000)
    parser.add_argument('-lower', type=bool, default=False)
    parser.add_argument('-output', type=str, default='../datas/paranmt-350k/train.parse_vocab')
    parser.add_argument('-cut_freq', type=int, default=0)
    parser.add_argument('-stopword', type=bool, default=False)
    args = parser.parse_args()
    main(args)
