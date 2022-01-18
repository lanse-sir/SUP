import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
from autocg.load_file import save_to_file, load_sentence

import argparse
import re

def contain_chinese(sentence):
    zhmodel = re.compile(u'[\u4e00-\u9fa5]')  #检查中文
    #zhmodel = re.compile(u'[^\u4e00-\u9fa5]')  #检查非中文
    match = zhmodel.search(sentence)
    if match:
        return True
    else:
        return False


def main(args):
    sents = load_sentence(args.src)
    parses = load_sentence(args.tgt)
    templates = load_sentence(args.temp)
    assert len(sents) == len(parses) == len(templates)

    new_sents = []
    new_parses = []
    new_templates = []
    for sent, parse, template in zip(sents, parses, templates):
        if contain_chinese(sent) or contain_chinese(parse):
            print(sent)
            print(parse)
            continue
            
        src_lengths = len(sent.split())
        tgt_lengths = len(parse.split())
        if src_lengths > args.max_length or src_lengths < args.min_length or tgt_lengths > args.max_length or tgt_lengths < args.min_length:
            continue
        new_sents.append(sent)
        new_parses.append(parse)
        new_templates.append(template)

    print('The filter number: ', len(sents) - len(new_sents))

    save_to_file(new_sents, args.out_src)
    save_to_file(new_parses, args.out_tgt)
    save_to_file(new_templates, args.out_temp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='../datas/dev_src.txt')
    parser.add_argument('--tgt', type=str, default='../datas/dev_tgt.txt')
    parser.add_argument('--temp', type=str, default='../datas/dev_src.pos')
    parser.add_argument('--min_length', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=30)
    parser.add_argument('--out_src', type=str, default='../datas/paranmt-350k-4~30/dev_src.txt')
    parser.add_argument('--out_tgt', type=str, default='../datas/paranmt-350k-4~30/dev_tgt.txt')
    parser.add_argument('--out_temp', type=str,
                        default='../datas/paranmt-350k-4~30/dev_src.pos')
    args = parser.parse_args()
    main(args)
