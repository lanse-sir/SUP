import argparse
from nltk.tree import Tree
import re
import os
import random

def list_to_file(filename, sents):
    with open(filename, 'w', encoding='utf-8') as f_w:
        for sent in sents:
            f_w.write(sent + '\n')


def extract_parses(fname):
    # extract parses from corenlp output
    # based on https://github.com/miyyer/scpn/blob/master/read_paranmt_parses.py
    with open(fname, 'r', encoding='utf-8') as f:
        count = 0
        sentences = []
        data = {'tokens': [], 'pos': [], 'parse': '', 'deps': []}
        for idx, line in enumerate(f):
            if idx <= 1:
                continue
            if line.startswith('Sentence #'):
                new_sent = True
                new_pos = False
                new_parse = False
                new_deps = False
                if idx == 2:
                    continue

                sentences.append(data)
                count += 1

                data = {'tokens': [], 'pos': [], 'parse': '', 'deps': []}

            # read original sentence
            elif new_sent:
                new_sent = False
                new_pos = True

            elif new_pos and line.startswith("Tokens"):
                continue

            # read POS tags
            elif new_pos and line.startswith('[Text='):
                line = line.strip().split()
                w = line[0].split('[Text=')[-1]
                pos = line[-1].split('PartOfSpeech=')[-1][:-1]
                data['tokens'].append(w)
                data['pos'].append(pos)

            # start reading const parses
            elif (new_pos or new_parse) and len(line.strip()):
                if line.startswith("Constituency parse"):
                    continue
                new_pos = False
                new_parse = True
                data['parse'] += ' ' + line.strip()

            # start reading deps
            elif (new_parse and line.strip() == "") or \
                    line.startswith("Dependency Parse"):
                new_parse = False
                new_deps = True

            elif new_deps and len(line.strip()):
                line = line.strip()[:-1].split('(', 1)
                rel = line[0]
                x1, x2 = line[1].split(', ')
                x1 = x1.replace("'", "")
                x2 = x2.replace("'", "")
                x1 = int(x1.rsplit('-', 1)[-1])
                x2 = int(x2.rsplit('-', 1)[-1])
                data['deps'].append((rel, x1 - 1, x2 - 1))

            else:
                new_deps = False

        sentences.append(data)

    return sentences


def deleaf(parse_string):
    tree = Tree.fromstring(parse_string.strip(), read_leaf=lambda s: "")
    for sub in tree.subtrees():
        for n, child in enumerate(sub):
            if isinstance(child, str):
                continue
            if len(list(child.subtrees(filter=lambda x: x.label() == '-NONE-'))) == len(child.leaves()):
                del sub[n]
    oneline = tree.pformat(margin=10000, parens=[" ( ", " ) "])
    oneline = re.sub(' +', ' ', oneline)
    return oneline

def load_parse(path):
    sentences = []
    if os.path.isfile(path):
        sentences = extract_parses(path)
    else:
        filelist = os.listdir(path)
        filelist = sorted(filelist)
        print("file nums ", len(filelist))
        for idx, file in enumerate(filelist):
            load_sents = extract_parses(os.path.join(path, file))
            print('file {} : num is {}'.format(file, len(load_sents)))
            sentences += load_sents
    return sentences

def train_dev_split(data, dev_size=2000):
    random.shuffle(data)
    train = data[:-dev_size]
    dev = data[-dev_size:]
    return train, dev

def save_to_file(sentences, args):
    all_poses = []
    all_parses = []
    all_sents = []
    for sent in sentences:
        if args.remove_word:
            parse = deleaf(sent['parse'])
        else:
            parse = sent['parse']
        all_poses.append((' '.join(sent['pos'])))
        all_parses.append(parse)
        all_sents.append(' '.join(sent['tokens']))
    print('Nums :', len(all_parses))
    if args.sent is not None:
        print('Sentence file is: ', args.sent)
        list_to_file(args.sent, all_sents)
    if args.pos is not None:
        print('Pos save file is: ', args.pos)
        list_to_file(args.pos, all_poses)
    if args.parse is not None:
        print('Parse save file is: ', args.parse)
        list_to_file(args.parse, all_parses)

def main(args):
    sentences = load_parse(args.f)
    print('total nums ',len(sentences))
    #train, dev = train_dev_split(sentences)
    #exit()
    save_to_file(sentences, args)
    #save_to_file(dev, dataset='dev')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract parse to file .')
    parser.add_argument('-f', type=str, help="a data dir or file.")
    # parser.add_argument('-s', type=str)
    parser.add_argument('-sent',type=str, help='the save file of sentence .')
    parser.add_argument('-pos', type=str, help="the save file of pos sequence.")
    parser.add_argument('-parse',type=str, help= "the save file of parse .")
    parser.add_argument('-remove_word', action='store_true')
    args = parser.parse_args()
    #print(args.pos)
    main(args)