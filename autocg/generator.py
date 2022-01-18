import argparse
import torch
from autocg.model import atcg
from autocg.train import evaluate
from autocg.load_file import load_file_sent_or_parser
from autocg.utils_file import extract_templates, string_to_list

model_file = 'checkpoint/model'
sents_data_fname = 'parser-data/ori_sent'
# parse_data_fname = 'data/test_parse_data'
parser = argparse.ArgumentParser()
parser.add_argument('-sent_pred', type=str, default='predict_result/sent.txt')
parser.add_argument('-parse_pred', type=str, default='predict_result/parse.txt')
opt = parser.parse_args()


def input_pair(sents, templates):
    i = []
    for s in sents:
        for t in templates:
            i.append((s, t))
    return i


templates = [
    '( ROOT ( S ( NP ) ( VP ) ( . ) ) )',
    '( ROOT ( S ( NP ) ( ADVP ) ( VP ) ( . ) ) )',
    '( ROOT ( S ( PP ) ( , ) ( NP ) ( VP ) ( . ) ) )',
    '( ROOT ( S ( `` ) ( S ) ( , ) ( '' ) ( NP ) ( VP ) ( . ) ) )',
    '( ROOT ( S ( CC ) ( NP ) ( VP ) ( . ) ) )',
    '( ROOT ( S ( S ) ( , ) ( NP ) ( VP ) ( . ) ) )',
    '( ROOT ( S ( S ) ( CC ) ( S ) ( . ) ) )',
    '( ROOT ( S ( S ) ( , ) ( CC ) ( S ) ( . ) ) )',
    '( ROOT ( S ( ADVP ) ( , ) ( NP ) ( VP ) ( . ) ) )',
    '( ROOT ( S ( VP ) ( . ) ) )',
    '( ROOT ( S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )'
]

sents = load_file_sent_or_parser(sents_data_fname, "sent")
# parses = load_file_sent_or_parser(parse_data_fname, "parse")
# template = extract_templates(parses)

if isinstance(templates[0], str):
    string_to_list(templates)
inputs = input_pair(sents, templates)
params = torch.load(model_file, map_location=lambda storage, loc: storage)
autocg_model = atcg(params['args'], params['word_vocab'], params['parse_vocab'])
autocg_model.load_state_dict(params['state_dict'])
if torch.cuda.is_available():
    autocg_model = autocg_model.cuda()
sent_dev_bleu, sent_preds, parse_preds = evaluate(model=autocg_model, test_data=inputs, word_vocab=params['word_vocab'],
                                                  parser_vocab=params['parse_vocab'], model_args=params['args'],
                                                  opt=opt, batch_size=4)
print(sent_dev_bleu)
# print(parse_dev_bleu)
for sent in sent_preds:
    print(sent)

for parse in parse_preds:
    print(parse)
