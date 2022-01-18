from autocg.train_classifier import data_to_idx
import torch
from autocg.sent_classifier import classifier


def load_sentence(path):
    instances = []
    with open(path, 'r', encoding='utf-8') as f_r:
        for line in f_r:
            sent, patt = line.strip().split('*')
            instances.append(((sent.split(), int(patt))))
    return instances


test_file = 'data/test/input.txt'
model_file = 'checkpoint/classifer_model'
test_sents = load_sentence(test_file)
params = torch.load(model_file, map_location=lambda storage, loc: storage)
model = classifier(params['args'], params['vocab'])
model.load_state_dict(params['state_dict'])
if torch.cuda.is_available:
    model = model.cuda()

def batch_serialization(batch, src_vocab, if_train=True):
    batch_size = len(batch)
    sents = [pair[0] for pair in batch]
    # parsers = [pair[1] for pair in batch]
    sent_type = [pair[1] for pair in batch]
    # tgt_sents = [['<s>'] + pair[0] + ['</s>'] for pair in batch]
    # tgt_parsers = [['<s>'] + pair[1] + ['</s>'] for pair in batch]
    # to index
    sents_idx_seq = [data_to_idx(s, src_vocab) for s in sents]

    # tgt_sents_idx_seq = [data_to_idx(tgt_s, src_vocab) for tgt_s in tgt_sents]
    # parsers_idx_seq = [data_to_idx(parse, tgt_vocab) for parse in parsers]

    # tgt_parsers_idx_seq = [data_to_idx(parse, tgt_vocab) for parse in tgt_parsers]

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
    # tgt_seq_lengths, tgt_seq_tensor, tgt_mask = padding(tgt_sents_idx_seq)
    # src_parse_seq_lengths, src_parse_seq_tensor, src_parse_mask = padding(parsers_idx_seq)
    # tgt_parse_seq_lengths, tgt_parse_seq_tensor, tgt_parse_mask = padding(tgt_parsers_idx_seq)
    # from big to small sort.
    src_seq_lengths, src_perm_idx = src_seq_lengths.sort(0, descending=True)
    # src_parse_seq_lengths, src_parse_perm_idx = src_parse_seq_lengths.sort(0, descending=True)
    src_seq_tensor = src_seq_tensor[src_perm_idx]
    src_mask = src_mask[src_perm_idx]
    # tgt_seq_tensor = tgt_seq_tensor[src_perm_idx]
    # src_parse_seq_lengths = src_parse_seq_lengths[src_parse_perm_idx]
    # src_parse_seq_tensor = src_parse_seq_tensor[src_parse_perm_idx]
    # src_parse_mask = src_parse_mask[src_parse_perm_idx]
    # tgt_parse_seq_tensor = tgt_parse_seq_tensor[src_parse_perm_idx]

    # sentence type :
    sent_type_tensor = torch.LongTensor(sent_type)[src_perm_idx]
    _, sent_seq_recover = src_perm_idx.sort(0, descending=False)
    # _, parse_seq_recover = src_parse_perm_idx.sort(0, descending=False)
    return src_seq_lengths, src_seq_tensor, src_mask, sent_seq_recover, sent_type_tensor


def evaluate(model, dev_data, vocab, batch_size=1):
    model.eval()
    right_instance = 0
    with torch.no_grad():
        for start in range(0, len(dev_data), batch_size):
            batch = dev_data[start:start + batch_size]
            src_seq_lengths, src_seq_tensor, src_mask, sent_seq_recover, sent_type_tensor = list(
                map(lambda x: x.cuda(), batch_serialization(batch,
                                                            vocab,
                                                            True)))
            _, logit, att = model(src_seq_tensor=src_seq_tensor, src_seq_mask=src_mask, src_seq_length=src_seq_lengths)
            score, idx = logit.topk(1)
            right_instance += torch.sum(idx.squeeze(1) == sent_type_tensor).item()
    return right_instance / len(dev_data)


acc = evaluate(model, test_sents, vocab=params['vocab'])
print(acc)