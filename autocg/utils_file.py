# from load_file import load_sentence
from nltk.tree import Tree
import re


# from nltk import ParentedTree

# extract the top two layer tree .
# parsers = load_sentence('data/test_parse_data')
def list_to_file(filename, sents):
    with open(filename, 'w', encoding='utf-8') as f_w:
        for sent in sents:
            f_w.write(sent + '\n')


def extract_top_layer_tree(t, level, mlevel):
    if level == mlevel:
        for idx, sub_t in enumerate(t):
            # print(type(sub_t))
            # extract_top_layer_tree(sub_t, level+1, mlevel)
            if isinstance(sub_t, Tree):
                t[idx] = Tree(sub_t.label(), [])
    else:
        for n in t:
            extract_top_layer_tree(n, level + 1, mlevel)


def tree_to_string(tree, node_list):
    node_list.append('(')
    node_list.append(tree.label())
    # node_list.append(')')
    for sub_tree in tree:
        tree_to_string(sub_tree, node_list)
        # node_list.append(')')
    node_list.append(')')


def extract_templates(parsers, height, remove_root=False):
    templates = []
    # height -= 2 # input height high 2 .
    for p in parsers:
        templates.append(extract_template(p, height, remove_root))
    return templates


def extract_template(p, l_level, remove_root=False):
    if remove_root:
        p = p.split()[2:-1]
        p = ' '.join(p)
    con_tree = Tree.fromstring(p)
    extract_top_layer_tree(con_tree, 0, l_level)
    # tree_s = con_tree.pformat()
    # return tree_s
    # print(con_tree)
    oneline = con_tree.pformat(margin=10000, parens=[" ( ", " ) "])
    oneline = re.sub(' +', ' ', oneline)
    return oneline


def strings_to_list(list_s):
    for idx, s in enumerate(list_s):
        list_s[idx] = s.split()


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
            if line.startswith('Document: '):
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


def get_parse(parses):
    parsers = []
    for parse in parses:
        parse_list = deleaf(parse).split()
        if 'TOP' or 'ROOT' in parse_list:
            parse_list = parse_list[2:-1]
        parsers.append(' '.join(parse_list))
    return parsers


def score_to_N(word):
    pattern = re.compile(r"\d+\\/\d+")
    matchs = re.findall(pattern, word)
    if word in matchs:
        return 'N'
    else:
        return word


def time_to_N(word):
    pattern = re.compile(r"\b\d+:\d+")
    matches = re.search(pattern, word)
    if matches is not None:
        matches = matches.group()
    else:
        matches = ''

    if matches == word:
        return 'N'
    else:
        return word


def num_to_N(word):
    pattern = re.compile(r"\b\d+,?\d*\.?\d*\b")
    matchs = re.findall(pattern, word)
    if word in matchs:
        return 'N'
    else:
        return word


def big_num_to_N(word):
    pattern = re.compile(r"\b\d+(,\d+)*\b")
    matchs = re.search(pattern, word)
    if matchs is not None:
        matchs = matchs.group()
    else:
        matchs = ''
    # print(type(matchs))
    if word == matchs:
        return 'N'
    else:
        return word


def word_to_N(word):
    word = num_to_N(word)
    word = score_to_N(word)
    word = time_to_N(word)
    word = big_num_to_N(word)
    return word


def tree_to_sentences(parses, word_to_n=False):
    sents = []
    for parse in parses:
        sent = []
        parse = Tree.fromstring(parse)
        for t in parse.pos():
            if t[1] != '-NONE-':
                word = t[0].lower()
                if word_to_n:
                    word = word_to_N(word)
                sent.append(word)
        sents.append(sent)
    return sents


def template_match(ref_p, pred_p):
    ref_temp = extract_template(ref_p, 1)  # default the top two level .
    pred_temp = extract_template(pred_p, 1)
    # print(ref_temp)
    if pred_temp == ref_temp:
        return 1
    else:
        return 0
