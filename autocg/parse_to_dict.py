import argparse
import sys

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

from nltk.tree import Tree
from autocg.load_file import load_sentence, save_to_file
from autocg.subword_nmt import apply_bpe
from tqdm import tqdm
import json


def cleanbrackets(string):
    a = {
        '-LRB-': '(',
        '-RRB-': ')',
        '-LSB-': '[',
        '-RSB-': ']',
        '-LCB-': '{',
        '-RCB-': '}',

        '-lrb-': '(',
        '-rrb-': ')',
        '-lsb-': '[',
        '-rsb-': ']',
        '-lcb-': '{',
        '-rcb-': '}',
    }
    try:
        return a[string]
    except:
        return string


def bpe_encode(word):
    if word in word2bpe:
        return word2bpe[word]

    bpe_segments = bpe_encoder.segment(word)
    word2bpe[word] = bpe_segments
    return bpe_segments


def dfs(root, s, bpe=False):
    # label = root._label.split()[0].replace('-','')
    label = root.label()
    # label = re.findall(r'-+[A-Z]+-|[A-Z]+\$|[A-Z]+|\.', root._label)[0]
    tree_dict = {label: []}
    # print(leaf, root._label)
    # if len(root._label.split()) > 1:
    #     tree_dict[label].append(root._label.split()[1])

    for child in root:
        if type(child) is str:
            if bpe:
                tree_dict[label] = bpe_encode(cleanbrackets(child))
            else:
                tree_dict[label] = cleanbrackets(child)
            return tree_dict
        # if child._label.split()[]
        else:
            tree_dict[label].append(dfs(child, s, bpe))
    return tree_dict


def dfsopti(tree, root, tree_dict, label, cnt):
    tree_dict[label] = []

    if type(tree[root]) is str:
        tree_dict[label].append(tree[root])
    else:
        for child in tree[root]:
            if type(child) is dict:
                child_label = list(child.keys())[0]
                if child_label not in cnt:
                    cnt[child_label] = 0
                cnt[child_label] += 1
                child_label = child_label + '-' + str(cnt[child_label])
                tree_dict[label].append(child_label)
                dfsopti(child, list(child.keys())[0], tree_dict, child_label, cnt)


def trees_to_dict(parses):
    parse_dicts = []

    pos_seqs = []
    word_seqs = []

    for i in tqdm(range(len(parses))):
        p_tree = Tree.fromstring(parses[i])
        words, pos = zip(*p_tree.pos())
        word_seqs.append(" ".join(words))
        pos_seqs.append(" ".join(pos))

        if args.to_dict:
            p_dict = dfs(p_tree, '', args.bpe)
            tree = p_dict
            tree_dict = {}
            cnt = {'ROOT': 0}
            dfsopti(tree, 'ROOT', tree_dict, 'ROOT', cnt)
            str_dict = json.dumps(tree_dict)
            parse_dicts.append(str_dict)
    return parse_dicts, word_seqs, pos_seqs


def main(args):
    print(args.input)
    # print(tgt_file)
    src_parses = load_sentence(args.input)
    print("Nums: ", len(src_parses))
    parse_dicts, sentences, pos_seqs = trees_to_dict(src_parses)

    if args.to_dict:
        print(f"Save to {args.output}. ")
        save_to_file(parse_dicts, args.output)

    if args.pos is not None:
        save_to_file(pos_seqs, args.pos)

    if args.word is not None:
        save_to_file(sentences, args.word)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--to_dict', action='store_true')
    parser.add_argument('--output', type=str)
    parser.add_argument('--pos', type=str)
    parser.add_argument('--word', type=str)

    parser.add_argument('--bpe', action='store_true')
    parser.add_argument('--bpe_vocab_file', type=str, default=None)
    parser.add_argument('--bpe_code_file', type=str, default=None)
    parser.add_argument('--threshold', type=int, default=None)
    args = parser.parse_args()
    if args.bpe_vocab_file is not None:
        with open(args.bpe_vocab_file, 'r') as f:
            voc = f.read().split('\n')
            if voc[-1].strip() == '':
                voc = voc[:-1]
            vocab = apply_bpe.read_vocabulary(voc, args.threshold)
            print('Vocab Size = {}'.format(len(vocab)))
    else:
        vocab = None

    if args.bpe_code_file is not None:
        codefile = open(args.bpe_code_file)
        bpe_encoder = apply_bpe.BPE(codefile, vocab=vocab)
        word2bpe = {}
    main(args)
