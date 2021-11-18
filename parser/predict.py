import sys
sys.path.append('../../../')
sys.path.append('../../')
import torch
import codecs
from cvae_model.classifer.model_seq2seq import seq2seq
from cvae_model.classifer.trainer import evaluate
from autocg.load_file import load_file_sent_or_parser
from autocg.subwordnmt.apply_bpe import BPE, read_vocabulary
# from autocg.utils_file import string_to_list
import argparse


parser = argparse.ArgumentParser(
    description="Training Syntactic Text Generation Models",
    usage="generator.py [<args>] [-h | --help]")
parser.add_argument('--label',type=str, default='../../data/para50w/test/test_tgt_template.txt')
parser.add_argument('--input',type=str,default='../../data/para50w/test/test_tgt_sent.txt')
parser.add_argument('--sent', type=str)
parser.add_argument('--model_file', type=str, default='models/model_15')
parser.add_argument('--eval_bs', type=int, default=10)
parser.add_argument('--greedy',type=bool, default=True)
parser.add_argument('--beam_size', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--bpe_codes_path', type=str, default='../../data/para5m/train/vocab/code_file_32k.txt')
parser.add_argument('--bpe_vocab_path', type=str, default = '../../data/para5m/train/vocab/vocab_32k.txt')
parser.add_argument('--bpe_vocab_thresh', type=int, default = 50)
opt = parser.parse_args()


# load bpe codes
bpe_codes = codecs.open(opt.bpe_codes_path, encoding='utf-8')
bpe_vocab = codecs.open(opt.bpe_vocab_path, encoding='utf-8')
bpe_vocab = read_vocabulary(bpe_vocab, opt.bpe_vocab_thresh)
bpe = BPE(bpe_codes, '@@', bpe_vocab, None)

sents = load_file_sent_or_parser(opt.input, "parse")
sents = [bpe.segment(sent).split() for sent in sents]

parses = load_file_sent_or_parser(opt.label, "sent")

inputs = list(zip(sents, parses))
print(inputs[0])
torch.cuda.set_device(opt.gpu)
params = torch.load(opt.model_file, map_location=lambda storage, loc: storage)
autocg_model = seq2seq(params['args'], params['word_vocab'], params['parse_vocab'])
autocg_model.load_state_dict(params['state_dict'])
if torch.cuda.is_available():
    autocg_model = autocg_model.cuda()
acc = evaluate(model=autocg_model, test_data=inputs, word_vocab=params['word_vocab'], parse_vocab=params['parse_vocab'], opt=opt)

print('Accuracy is %.3f'%(acc))
print('done')
# print(parse_dev_bleu)
# for sent in sent_preds:
#     print(sent)

# for parse in parse_preds:
#     print(parse)
