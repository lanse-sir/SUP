import sys
sys.path.append('../../')
sys.path.append('../')
import torch
from train_classifier import binary_classifer, load_sentiment_label,data_to_idx
import argparse


def load_processed_file(filename):
    groups = []
    group = []
    with open(filename, 'r', encoding='utf-8') as f_r:
        for line in f_r:
            if line.startswith('--------------'):
                groups.append(group)
                group = []
            else:
                group.append(line.strip().split())
    return groups


parser = argparse.ArgumentParser(
    description="Training Syntactic Text Generation Models",
    usage="generator.py [<args>] [-h | --help]")
parser.add_argument('--input', type=str, default='../data/')
parser.add_argument('--label',type=str, default='data/remain_dev_label.txt')
parser.add_argument('--pred_label', type=str)
parser.add_argument('--model_file', type=str)
opt = parser.parse_args()


def batch_serialization(batch, src_vocab, if_train=False):
    batch_size = len(batch)
    sents = [pair for pair in batch]
    sents_idx_seq = [data_to_idx(s, src_vocab) for s in sents]
    #print(sents_idx_seq)
    # as for src sentence .
    def padding(sents):
        word_seq_lengths = torch.LongTensor(list(map(len, sents)))
        max_seq_len = word_seq_lengths.max().item()
        word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
        # tgt_word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
        mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).byte()
        for idx, (seq, seqlen) in enumerate(zip(sents, word_seq_lengths)):
            seqlen = seqlen.item()
            word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
            mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
        return word_seq_lengths, word_seq_tensor, mask

    src_seq_lengths, src_seq_tensor, src_mask = padding(sents_idx_seq)
    # tgt_parse_seq_lengths, tgt_parse_seq_tensor, tgt_parse_mask = padding(tgt_parsers_idx_seq)
    # from big to small sort.
    src_seq_lengths, src_perm_idx = src_seq_lengths.sort(0, descending=True)
    src_seq_tensor = src_seq_tensor[src_perm_idx]
    src_mask = src_mask[src_perm_idx]
    # sentence type :
    _, sent_seq_recover = src_perm_idx.sort(0, descending=False)
    return src_seq_lengths, src_seq_tensor, src_mask, sent_seq_recover


def evaluate(model, dev_data, vocab):
    model.eval()
    pred_res = []
    #right_instance = 0
    with torch.no_grad():
        for group in dev_data:
            batch = group
            src_seq_lengths, src_seq_tensor, src_mask, sent_seq_recover = list(
                map(lambda x: x.cuda(), batch_serialization(batch,
                                                            vocab)))
            inputs = {'src tensor': src_seq_tensor, 'src lengths': src_seq_lengths, ' src mask': src_mask,
                      'seq recover': sent_seq_recover}
            logits = model.predict(inputs)
            score, idx = logits.topk(1)
            pred_res.append(idx.squeeze(1).cpu().numpy().tolist())
    return pred_res


def dev_broken(pred_res, labels):
    generate_para = 0
    broken = 0
    adv_exam_group_idx = []
    assert len(pred_res) == len(labels)
    for idx, group in enumerate(pred_res):
        if len(group)>1:
            adv_exam_sent_idx = []
            generate_para += 1
            for j, label in enumerate(group):
                if j == 0:
                    continue
                if label != labels[idx] and group[0]==labels[idx]:
                    broken += 1
                    break
            for j, label in enumerate(group):
                if j == 0:
                    #adv_exam_group_idx.append(idx)
                    adv_exam_sent_idx.append(j)
                elif label != labels[idx] and group[0]==labels[idx]:
                    adv_exam_sent_idx.append(j)
            adv_exam_group_idx.append((idx,adv_exam_sent_idx))
    broken_ratio = broken/generate_para
    return generate_para, broken, broken_ratio, adv_exam_group_idx
    
    
sents = load_processed_file(opt.input)
labels = load_sentiment_label(opt.label)

print('Input File: ',opt.input)
print('Model File: ', opt.model_file)
print('Save Sentence Path: ',opt.pred_label)
print("Input Size: ", len(sents))

print(sents[0])
params = torch.load(opt.model_file, map_location=lambda storage, loc: storage)
model = binary_classifer(params['args'], params['word_vocab'])
model.load_state_dict(params['state_dict'])
if torch.cuda.is_available():
    model = model.cuda()
predict_results = evaluate(model=model, dev_data=sents, vocab=params['word_vocab'])
generate_para, broken, broken_ratio, adv_group_idx = dev_broken(predict_results, labels)
print("Multi Sentence: {}, Broken: {}, Broken Ratio: {}".format(generate_para, broken, broken_ratio))
print('done')

#assert len(labels)==len(predict_results)
with open(opt.pred_label,'w') as f_w:
    for e in adv_group_idx:
        if len(e[1]) > 1:
            for s in e[1]:
                f_w.write(' '.join(sents[e[0]][s])+'\t'+str(predict_results[e[0]][s])+'\n')
                #print(' '.join(sents[e[0]][s])+'\t'+str(predict_results[e[0]][s]))
            #print('-'*100)
            f_w.write('-'*100+'\n')
                  #for i in range(len(adv_group_idx)):
#    print(' '.join(sents[adv_group_idx[i]][0]))
#    print(' '.join(sents[adv_group_idx[i]][adv_sent_idx[i]]))
#    print('-'*100)
# print(parse_dev_bleu)
#for sent in sent_preds:
#    print(sent)

#for parse in parse_preds:
#    print(parse)
