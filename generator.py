import sys
import pickle
import codecs

sys.path.append('../../')
sys.path.append('../')
import torch
from model_cvae import model_cvae
from train_cvae import evaluate
from autocg.subwordnmt.apply_bpe import BPE, read_vocabulary
from autocg.load_file import load_file_sent_or_parser
# from autocg.utils_file import string_to_list
import argparse

parser = argparse.ArgumentParser(
    description="Training Syntactic Text Generation Models",
    usage="generator.py [<args>] [-h | --help]")
parser.add_argument('--dev_sent', type=str, default='../data/para50w/test/test_src_sent.txt')
parser.add_argument('--dev_parse', type=str, default='../data/para50w/test/test_tgt_template.txt')
parser.add_argument('--dev_ref', type=str, default='../data/para50w/test/test_tgt_sent.txt')
parser.add_argument('--sent', type=str)
parser.add_argument('--model_file', type=str)
parser.add_argument('--word_occur', type=str, default='../data/word_occurence.pkl')
parser.add_argument('--bow_ratio', type=float, default=1.0)
parser.add_argument('--tree_gru', action='store_true')
parser.add_argument('--ht', type=int, default=3)
parser.add_argument('--eval_bs', type=int, default=50)
parser.add_argument('--greedy', action='store_true')
parser.add_argument('--beam_size', type=int, default=4)
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--bpe', action='store_true')
parser.add_argument('--bpe_codes_path', type=str, default='../data/para5m/train/vocab/code_file_32k.txt')
parser.add_argument('--bpe_vocab_path', type=str, default = '../data/para5m/train/vocab/vocab_32k.txt')
parser.add_argument('--bpe_vocab_thresh', type=int, default = 50)
opt = parser.parse_args()

### load bpe code.
if opt.bpe:
    bpe_codes = codecs.open(opt.bpe_codes_path, encoding='utf-8')
    bpe_vocab = codecs.open(opt.bpe_vocab_path, encoding='utf-8')
    bpe_vocab = read_vocabulary(bpe_vocab, opt.bpe_vocab_thresh)
    bpe = BPE(bpe_codes, '@@', bpe_vocab, None)

    sents = load_file_sent_or_parser(opt.dev_sent, "parse")
    sents = [bpe.segment(sent).split() for sent in sents]
else:
    sents = load_file_sent_or_parser(opt.dev_sent, "sent")

parses = load_file_sent_or_parser(opt.dev_parse, "parse")
word_occurence, document_num = pickle.load(open(opt.word_occur, 'rb'))

inputs = list(zip(sents, parses))
print(inputs[0])
torch.cuda.set_device(opt.gpu)
params = torch.load(opt.model_file, map_location=lambda storage, loc: storage)
autocg_model = model_cvae(params['args'], params['word_vocab'], params['parse_vocab'])
autocg_model.load_state_dict(params['state_dict'])
if torch.cuda.is_available():
    autocg_model = autocg_model.cuda()
ori_bleu, ref_bleu = evaluate(model=autocg_model,
                              test_data=inputs,
                              word_vocab=params['word_vocab'],
                              parse_vocab=params['parse_vocab'],
                              word_occurence = word_occurence,
                              document_num = document_num,
                              opt=opt)

print('Ref BLEU=%.3f, Ori BLEU=%.3f' % (ref_bleu, ori_bleu))
print('done')
# print(parse_dev_bleu)
# for sent in sent_preds:
#    print(sent)

# for parse in parse_preds:
#    print(parse)
