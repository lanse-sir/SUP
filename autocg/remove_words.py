import sys

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
from tqdm import tqdm
from autocg.utils_file import extract_template, deleaf
from autocg.utils_file import tree_to_string
from load_file import list_to_file, load_file_sent_or_parser

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default=None)
parser.add_argument('-s', type=str, default=None)
args = parser.parse_args()
sentences = load_file_sent_or_parser(args.f, type='parse')

all_template = []
for i in tqdm(range(len(sentences))):
    p = deleaf(sentences[i])  # remove word leaf node.

    all_template.append(p)
list_to_file(args.s, all_template)

# python template.py -f data/para50w/train/train_parse.txt -s data/para50w/train/train_top3_parse.txt -min_level 1
