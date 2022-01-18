import argparse
import sys

sys.path.append('../')
from autocg.utils_file import extract_template
from autocg.load_file import load_sentence, list_to_file

parser = argparse.ArgumentParser(description=' Prune the parse tree. ')
parser.add_argument('-i', type=str, default='parser-data/test_y.txt')
parser.add_argument('-s', type=str)
args = parser.parse_args()

prune_trees = []
height = [3, 4, 5, 6, 7]
parses = load_sentence(args.i)

for p in parses:
    for h in height:
        h -= 2
        prune_trees.append(extract_template(p, h))

print('original parse num: ', len(parses))
print('prune trees num：', len(prune_trees))

list_to_file(args.s, prune_trees)
