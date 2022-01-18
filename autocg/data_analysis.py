import pandas as pd
from collections import Counter
from autocg.load_file import load_sentence


inp_file = "../datas/paranmt-350k-4~30/train/train_src_parse-4.txt"
parse_tree = load_sentence(inp_file)

parse = Counter(parse_tree)

print(len(parse))
for p, freq, in parse.most_common(10):
    print(p, "  ", freq)


