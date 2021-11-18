import sys
import codecs
sys.path.append('../../')
sys.path.append('../')
import torch
import pickle
from model_cvae import model_cvae
from train_cvae import generate_sentence
from autocg.subwordnmt.apply_bpe import BPE, read_vocabulary
from fine_tune import load_template
from autocg.load_file import load_file_sent_or_parser, save_to_file
from autocg.utils_file import string_to_list
import argparse

parser = argparse.ArgumentParser(
    description="Training Syntactic Text Generation Models",
    usage="generator.py [<args>] [-h | --help]")
parser.add_argument('--input_sent', type=str, default='../data/')
parser.add_argument('--input_template',type=str, default='../data/para50w/template-10.txt')
parser.add_argument('--sent', type=str, default='generate/test_sent')
parser.add_argument('--model_file', type=str, default='checkpoint/model')
parser.add_argument('--beam_size',type=int, default=4)
parser.add_argument('--eval_bs', type=int, default=100)
parser.add_argument('--bow_ratio', type=float, default=1.0)
parser.add_argument('--word_occur', type=str, default='../data/word_occurence.pkl')
parser.add_argument('--tree_gru', type=bool, default=True)
parser.add_argument('--greedy', type=bool, default=False)

parser.add_argument('--bpe', type=bool, default=True)
parser.add_argument('--bpe_codes_path', type=str, default='../data/para5m/train/vocab/code_file_32k.txt')
parser.add_argument('--bpe_vocab_path', type=str, default = '../data/para5m/train/vocab/vocab_32k.txt')
parser.add_argument('--bpe_vocab_thresh', type=int, default = 50)
opt = parser.parse_args()

word_occurence, document_num = pickle.load(open(opt.word_occur, 'rb'))

def input_pair(sents, templates):
    i = []
    for s in sents:
        for t in templates:
            i.append((s, t))
    return i

if opt.bpe:
    bpe_codes = codecs.open(opt.bpe_codes_path, encoding='utf-8')
    bpe_vocab = codecs.open(opt.bpe_vocab_path, encoding='utf-8')
    bpe_vocab = read_vocabulary(bpe_vocab, opt.bpe_vocab_thresh)
    bpe = BPE(bpe_codes, '@@', bpe_vocab, None)
    
    sents = load_file_sent_or_parser(opt.input_sent, "parse")
    sents = [bpe.segment(sent).split() for sent in sents]
else:  
    sents = load_file_sent_or_parser(opt.input_sent, "sent")
    
parses = load_template(opt.input_template)
print('Input File: ',opt.input_sent)
print('Template File: ', opt.input_template)
print("The Number of Template: ",len(parses))
print('Model File: ', opt.model_file)
print('Save Sentence Path: ',opt.sent)
inputs = input_pair(sents, parses)
print("Input Size: ", len(inputs))
print(inputs[0])
params = torch.load(opt.model_file)
autocg_model = model_cvae(params['args'], params['word_vocab'], params['parse_vocab'])
autocg_model.load_state_dict(params['state_dict'])

if torch.cuda.is_available():
    autocg_model = autocg_model.cuda()
generations = generate_sentence(model=autocg_model, 
                                test_data=inputs, 
                                word_vocab=params['word_vocab'], 
                                parse_vocab=params['parse_vocab'],
                                word_occurence = word_occurence,
                                document_num = document_num,
                                opt=opt)

res = [' '.join(s).replace('@@ ', '') for s in generations]
print('Save to {}'.format(opt.sent))
save_to_file(res, opt.sent)
print('done')
# print(parse_dev_bleu)
#for sent in sent_preds:
#    print(sent)

#for parse in parse_preds:
#    print(parse)
