from nltk.tree import Tree
import json
import re


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


def save_parse_corpus(fname, parser_list):
    with open(fname, 'w', encoding='utf-8') as f_w:
        for line in parser_list:
            f_w.write(line + '\n')


def load_file(corpus):
    parsers = list()
    sents = list()
    with open(corpus, 'r', encoding='utf-8') as f_r:
        for line in f_r:
            content_dict = json.loads(line.strip().split('\t')[0])
            parse = content_dict['parse']
            sents.append(' '.join(content_dict['tokens']))
            parsers.append(deleaf(parse))
    return parsers, sents


def list_to_file(filename, sents):
    with open(filename, 'w', encoding='utf-8') as f_w:
        for sent in sents:
            f_w.write(sent + '\n')


parser_list, sents = load_file('F:/NLPcorpus/sent_pattern/reference/itis.txt')
save_parse_corpus('data/parser.txt', parser_list)
list_to_file('data/itis.txt', sents)
