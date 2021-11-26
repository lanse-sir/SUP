import torch
from zss import Node
import json
import numpy as np
import random
from torch.autograd import Variable
from autocg.content_recognition import select_content_words
from nltk.tree import Tree


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def list_2_batch(list_tensor, if_train=False):
    batch_size = len(list_tensor)
    leave_lengths = torch.LongTensor(list(map(len, list_tensor)))
    max_lengths = leave_lengths.max().item()
    dim = list_tensor[0][0].size(0)
    leave_seq_tensor = torch.zeros((batch_size, max_lengths, dim), requires_grad=if_train)
    mask = torch.zeros((batch_size, max_lengths), requires_grad=if_train).byte()
    for i in range(batch_size):
        seq_len = leave_lengths[i].item()
        leave_seq_tensor[i, :seq_len] = torch.stack(list_tensor[i])
        mask[i, :seq_len] = torch.Tensor([1] * seq_len)
    leave_lengths, idx_perm = leave_lengths.sort(0, descending=True)
    leave_seq_tensor = leave_seq_tensor[idx_perm]
    _, idx_recover = idx_perm.sort(0, descending=False)
    return to_variable(leave_seq_tensor), to_variable(leave_lengths), to_variable(
        mask), to_variable(idx_recover)


def pack_batch(sents, templates):
    parses = []
    for i in range(len(sents)):
        parses.append(random.choice(templates))
    batch = list(zip(sents, parses))
    return batch


def batch_process(sents, if_train=True):
    batch_size = len(sents)
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


def bow_onehot(sents, word_vocab, word_occurence, document_num, opt):
    # Target sentence bag of word label .
    batch_size = len(sents)
    contents = select_content_words(sents, word_occurence, document_num, ratio=opt.bow_ratio)
    contents_id = [[word_vocab[w] for w in sent if w in word_vocab] for sent in contents]
    assert len(contents_id) == batch_size
    bag_label = torch.zeros((batch_size, len(word_vocab))).float()
    for idx in range(batch_size):
        bag_label[idx].scatter_(-1, torch.LongTensor(contents_id[idx]), 1.)
    return bag_label, contents_id


def data_to_idx(sent, vocab):
    return [vocab.get(w, vocab['<unk>']) for w in sent]


def batch_serialization_parallel(batch, word_vocab, parse_vocab, opt, word_occurence=None, document_num=None,
                                 if_train=True, mode='decode'):
    batch_size = len(batch)

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

    # sentence process.
    sents = [pair[0] for pair in batch]
    sents_idx_seq = [data_to_idx(s, word_vocab) for s in sents]

    src_seq_lengths, src_seq_tensor, src_mask = padding(sents_idx_seq)
    src_seq_lengths, src_perm_idx = src_seq_lengths.sort(0, descending=True)
    src_seq_tensor = src_seq_tensor[src_perm_idx]
    src_mask = src_mask[src_perm_idx]
    _, sent_seq_recover = src_perm_idx.sort(0, descending=False)

    # parse tree process.
    # string sequence.
    if opt.tree_gru:
        parsers = [pair[1] for pair in batch]
        tgt_trees = trees_to_dict(parsers)

        # remove_leaves_from_trees(parsers, bpe=False, pos=False)
        # tgt_trees, tgt_phrase_starts = trim_trees(parsers, opt.ht)

        # template = []
        # deep_first_travel(template, tgt_trees[0], 'ROOT')

    else:
        parsers = [pair[1] for pair in batch]
        parsers_idx_seq = [data_to_idx(parse, parse_vocab) for parse in parsers]

        src_parse_seq_lengths, src_parse_seq_tensor, src_parse_mask = padding(parsers_idx_seq)
        src_parse_seq_lengths, src_parse_perm_idx = src_parse_seq_lengths.sort(0, descending=True)
        src_parse_seq_tensor = src_parse_seq_tensor[src_parse_perm_idx]
        src_parse_mask = src_parse_mask[src_parse_perm_idx]
        _, parse_seq_recover = src_parse_perm_idx.sort(0, descending=False)

    # class parse tree: classifer to decode the controlled syntax.
    if opt.tree_gru:
        # c_parse_seq = trees_travel(dict_trees, root='ROOT')
        c_parse_seq = [pair[1].split() for pair in batch]
    else:
        c_parse_seq = parsers
    c_tgt_parses = [['<s>'] + seq + ['</s>'] for seq in c_parse_seq]
    c_tgt_parses_idx = [data_to_idx(parse, parse_vocab) for parse in c_tgt_parses]
    c_tgt_parses_lengths, c_tgt_parses_seq, c_tgt_parses_mask = padding(c_tgt_parses_idx)

    # Target sentence bag of word label .
    # contents = select_content_words(sents, word_occurence, document_num, ratio=opt.bow_ratio)
    # contents_id = [data_to_idx(sent, word_vocab) for sent in contents]
    contents_id =  sents_idx_seq
    _, contents_seq_tensor, _ = padding(contents_id)

    #assert len(contents_id) == batch_size
    bag_label = torch.zeros((batch_size, len(word_vocab))).float()
    for idx in range(batch_size):
        bag_label[idx].scatter_(-1, torch.LongTensor(contents_id[idx]), 1.)

    if mode == 'train':
        tgt_sents = [['<s>'] + pair[2] + ['</s>'] for pair in batch]
        # tgt_parsers = [['<s>'] + pair[2] + ['</s>'] for pair in batch]
        # tgt_parsers_idx_seq = [data_to_idx(parse, parse_vocab) for parse in tgt_parsers]
        tgt_sents_idx_seq = [data_to_idx(tgt_s, word_vocab) for tgt_s in tgt_sents]
        # tgt_parse_seq_lengths, tgt_parse_seq_tensor, tgt_parse_mask = padding(tgt_parsers_idx_seq)
        tgt_seq_lengths, tgt_seq_tensor, tgt_mask = padding(tgt_sents_idx_seq)

    else:
        tgt_seq_tensor = torch.Tensor([0.])
        bag_label = torch.Tensor([0.])

    if torch.cuda.is_available():
        if opt.tree_gru:
            return {
                'src_lengths': src_seq_lengths.cuda(),
                'src_seq': src_seq_tensor.cuda(),
                'src_mask': src_mask.cuda(),
                'src_recover': sent_seq_recover.cuda(),
                'tree': tgt_trees,
                'tgt_sent': tgt_seq_tensor.cuda(),
                'c_tgt_parse': c_tgt_parses_seq.cuda(),
                'bow': bag_label.cuda(),
                'con_word': contents_seq_tensor.cuda()
            }
        else:
            return {
                'src_lengths': src_seq_lengths.cuda(),
                'src_seq': src_seq_tensor.cuda(),
                'src_mask': src_mask.cuda(),
                'src_recover': sent_seq_recover.cuda(),
                'parse_seq_lengths': src_parse_seq_lengths.cuda(),
                'parse_seq': src_parse_seq_tensor.cuda(),
                'parse_mask': src_parse_mask.cuda(),
                'parse_recover': parse_seq_recover.cuda(),
                'tgt_sent': tgt_seq_tensor.cuda(),
                'c_tgt_parse': c_tgt_parses_seq.cuda(),
                'bow': bag_label.cuda(),
                'con_word': contents_seq_tensor.cuda()
            }
    else:
        if opt.tree_gru:
            return {
                'src_lengths': src_seq_lengths,
                'src_seq': src_seq_tensor,
                'src_mask': src_mask,
                'src_recover': sent_seq_recover,
                'tree': tgt_trees,
                'tgt_sent': tgt_seq_tensor,
                'c_tgt_parse': c_tgt_parses_seq,
                'bow': bag_label,
                'con_word': contents_seq_tensor
            }
        else:
            return {
                'src_lengths': src_seq_lengths,
                'src_seq': src_seq_tensor,
                'src_mask': src_mask,
                'src_recover': sent_seq_recover,
                'parse_seq_lengths': src_parse_seq_lengths,
                'parse_seq': src_parse_seq_tensor,
                'parse_mask': src_parse_mask,
                'parse_recover': parse_seq_recover,
                'tgt_sent': tgt_seq_tensor,
                'c_tgt_parse': c_tgt_parses_seq,
                'bow': bag_label,
                'con_word': contents_seq_tensor
            }


def batch_sentence_tree(batch, word_vocab, ht=3, if_train=True):
    batch_size = len(batch)
    sents = [pair[0] for pair in batch]
    parsers = [json.loads(pair[1]) for pair in batch]

    sents_idx_seq = [data_to_idx(s, word_vocab) for s in sents]

    # parsers_idx_seq = [data_to_idx(parse, tgt_vocab) for parse in parsers]
    # tgt_idx_seq = [data_to_idx(tgt, src_vocab) for tgt in tgts]

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

    def att_to_onehot(atts):
        lengths = torch.LongTensor(list(map(len, atts)))
        max_seq_len = lengths.max().item()
        leave_lengths = [seq[-1] for seq in atts]
        max_leaves = max(leave_lengths)
        one_hot = torch.zeros((batch_size, max_seq_len, max_leaves))
        label_smoothing = 0.1
        for b in range(batch_size):
            lens = lengths[b].item()
            # for s in range(max_seq_len):
            assert leave_lengths[b] != 0  # ensure the fenmu is not zero.
            one_hot[b, :lens, :leave_lengths[b]] = torch.FloatTensor(
                [label_smoothing / leave_lengths[b]] * leave_lengths[b])
            one_hot[b, :lens].scatter_add_(-1, (torch.LongTensor(atts[b]) - 1).unsqueeze(1),
                                           torch.FloatTensor([1 - label_smoothing] * len(atts[b])).unsqueeze(1))
        return one_hot

    src_seq_lengths, src_seq_tensor, src_mask = padding(sents_idx_seq)
    src_seq_lengths, src_perm_idx = src_seq_lengths.sort(0, descending=True)
    src_seq_tensor = src_seq_tensor[src_perm_idx]
    _, src_sent_seq_recover = src_perm_idx.sort(0, descending=False)
    # _, src_parse_seq_recover = src_parse_perm_idx.sort(0, descending=False)
    # _, tgt_seq_recover = tgt_parse_perm_idx.sort(0, descending=False)

    remove_leaves_from_trees(parsers, True)
    tgt_trees, tgt_phrase_starts = trim_trees(parsers, ht)
    # print(tgt_phrase_starts)
    # tgt_phrase_span = for pair in batch
    if if_train:
        # tgt_sents = [['<s>'] + phrase_span(batch[i][2], tgt_phrase_starts[i]) + ['</s>'] for i in range(batch_size)]
        tgt_sents = [['<s>'] + batch[i][2] + ['</s>'] for i in range(batch_size)]
        tgt_idx_seq = [data_to_idx(s, word_vocab) for s in tgt_sents]
        tgt_seq_lengths, tgt_seq_tensor, tgt_sent_mask = padding(tgt_idx_seq)
        att_targets = []
        for i in range(batch_size):
            flag = 0
            t = []
            for j in range(len(tgt_sents[i]) - 1):  # don't need the <s>.
                if j in tgt_phrase_starts[i]:
                    flag += 1
                t.append(flag)
            att_targets.append(t)

        smooth_one_hot = att_to_onehot(att_targets)
    else:
        tgt_seq_tensor = torch.Tensor([0.])
        smooth_one_hot = torch.Tensor([0.])

    if torch.cuda.is_available():
        return {
            'src_lengths': src_seq_lengths.cuda(),
            'src_seq': src_seq_tensor.cuda(),
            'src_mask': src_mask.cuda(),
            'tgt_seq': tgt_seq_tensor.cuda(),
            'src_recover': src_sent_seq_recover.cuda(),
            'tgt_tree': tgt_trees,
            'att_label': smooth_one_hot.cuda()
        }
    else:
        return {
            'src_lengths': src_seq_lengths,
            'src_seq': src_seq_tensor,
            'src_mask': src_mask,
            'tgt_seq': tgt_seq_tensor,
            'src_recover': src_sent_seq_recover,
            'tgt_tree': tgt_trees,
            'att_label': smooth_one_hot
        }


def pointer_batch_sentence_tree(batch, word_vocab, opt, pos_tag=None, word_occurence=None, sents_num=1000000,
                                if_train=True):
    ht = opt.ht
    nvoc = len(word_vocab)
    batch_size = len(batch)
    sents = [pair[0] for pair in batch]
    parsers = [json.loads(pair[1]) for pair in batch]
    sents_content = select_content_words(sents, word_occurence, sents_num, ratio=0.4)

    def sent2id(sent, voc, nvoc):
        idx = []
        extend_idx = []
        oov = []
        for w in sent:
            if w in voc:
                idx.append(voc[w])
                extend_idx.append(voc[w])
            else:
                oov.append(w)
                extend_idx.append(nvoc + oov.index(w))
                idx.append(voc['<unk>'])
        return idx, extend_idx, oov

    def sents2id(sents, voc, nvoc):
        batch_idx = []
        batch_extend_idx = []
        batch_oov = []
        for sent in sents:
            idx, extend_idx, oov = sent2id(sent, voc, nvoc)
            batch_idx.append(idx)
            batch_extend_idx.append(extend_idx)
            batch_oov.append(oov)
        return batch_idx, batch_extend_idx, batch_oov

    # parsers_idx_seq = [data_to_idx(parse, tgt_vocab) for parse in parsers]
    # tgt_idx_seq = [data_to_idx(tgt, src_vocab) for tgt in tgts]
    def para2id(para, voc, nvoc, oov):
        inp_id = [voc['<s>']]
        tgt_id = []
        for w in para:
            if w in voc:
                idx = voc[w]
                inp_id.append(idx)
                tgt_id.append(idx)
            else:
                if w in oov:
                    inp_id.append(voc['<unk>'])
                    tgt_id.append(nvoc + oov.index(w))
                else:
                    inp_id.append(voc['<unk>'])
                    tgt_id.append(voc['<unk>'])
        tgt_id.append(voc['</s>'])
        assert len(inp_id) == len(tgt_id)
        return inp_id, tgt_id

    def paras2id(paras, voc, nvoc, oovs):
        inps_ids = []
        tgt_ids = []
        for i, para in enumerate(paras):
            inp_id, tgt_id = para2id(para, voc, nvoc, oovs[i])
            inps_ids.append(inp_id)
            tgt_ids.append(tgt_id)
        return inps_ids, tgt_ids

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

    def att_to_onehot(atts):
        lengths = torch.LongTensor(list(map(len, atts)))
        max_seq_len = lengths.max().item()
        leave_lengths = [seq[-1] for seq in atts]
        max_leaves = max(leave_lengths)
        one_hot = torch.zeros((batch_size, max_seq_len, max_leaves))
        label_smoothing = 0.1
        for b in range(batch_size):
            lens = lengths[b].item()
            # for s in range(max_seq_len):
            assert leave_lengths[b] != 0  # ensure the fenmu is not zero.
            one_hot[b, :lens, :leave_lengths[b]] = torch.FloatTensor(
                [label_smoothing / leave_lengths[b]] * leave_lengths[b])
            one_hot[b, :lens].scatter_add_(-1, (torch.LongTensor(atts[b]) - 1).unsqueeze(1),
                                           torch.FloatTensor([1 - label_smoothing] * len(atts[b])).unsqueeze(1))
        return one_hot

    sents_idx_seq, sents_extend_idx_seq, oovs = sents2id(sents, word_vocab, nvoc)
    src_seq_lengths, src_seq_tensor, src_mask = padding(sents_idx_seq)
    src_seq_lengths, src_perm_idx = src_seq_lengths.sort(0, descending=True)
    src_seq_tensor = src_seq_tensor[src_perm_idx]
    _, src_sent_seq_recover = src_perm_idx.sort(0, descending=False)
    src_extend_lengths, src_extend_tensor, _ = padding(sents_extend_idx_seq)

    # content aware part .
    if opt.use_content:
        contents_idx_seq, contents_extend_idx_seq, contents_oovs = sents2id(sents_content, word_vocab, nvoc)
        content_seq_lengths, content_seq_tensor, content_mask = padding(contents_idx_seq)
        content_seq_lengths, content_perm_idx = content_seq_lengths.sort(0, descending=True)
        content_seq_tensor = content_seq_tensor[content_perm_idx]
        _, content_seq_recover = content_perm_idx.sort(0, descending=False)
        content_extend_lengths, content_extend_tensor, _ = padding(contents_extend_idx_seq)
    else:
        content_seq_tensor = torch.Tensor([0.])
        content_seq_lengths = torch.Tensor([0.])
        content_mask = torch.Tensor([0.])
        content_seq_recover = torch.Tensor([0.])

    # oovs to seqence .
    max_oovs = max(len(oov) for oov in oovs)
    oov_zeros = torch.zeros((batch_size, max_oovs))

    # remove pos and leaves word .
    remove_pos = False
    if pos_tag is not None:
        remove_pos_from_trees(parsers, pos_tag)
        remove_pos = True
    remove_leaves_from_trees(parsers, bpe=False, pos=remove_pos)

    tgt_trees, tgt_phrase_starts = trim_trees(parsers, ht)
    if if_train:
        tgt_sents = [batch[i][2] for i in range(batch_size)]
        inps_idx, tgts_idx = paras2id(tgt_sents, word_vocab, nvoc, oovs)
        inp_seq_lengths, inp_seq_tensor, _ = padding(inps_idx)
        tgt_seq_lengths, tgt_seq_tensor, _ = padding(tgts_idx)
        assert inp_seq_tensor.size(1) == tgt_seq_tensor.size(1)
        att_targets = []
        for i in range(batch_size):
            flag = 0
            t = []
            for j in range(len(tgts_idx[i])):
                if j in tgt_phrase_starts[i]:
                    flag += 1
                t.append(flag)
            att_targets.append(t)
        smooth_one_hot = att_to_onehot(att_targets)
    else:
        inp_seq_tensor = torch.Tensor([0.])
        tgt_seq_tensor = torch.Tensor([0.])
        smooth_one_hot = torch.Tensor([0.])

    if torch.cuda.is_available():
        return {
            'src_lengths': src_seq_lengths.cuda(),
            'src_seq': src_seq_tensor.cuda(),
            'src_mask': src_mask.cuda(),
            'src_extend_seq': src_extend_tensor.cuda(),
            'oovs': oov_zeros.cuda(),
            'oov_list': oovs,
            'inp_seq': inp_seq_tensor.cuda(),
            'tgt_seq': tgt_seq_tensor.cuda(),
            'src_recover': src_sent_seq_recover.cuda(),
            'tgt_tree': tgt_trees,
            'att_label': smooth_one_hot.cuda(),
            'con_seq': content_seq_tensor.cuda(),
            'con_lengths': content_seq_lengths.cuda(),
            'con_mask': content_mask.cuda(),
            'con_recover': content_seq_recover.cuda()
        }
    else:
        return {
            'src_lengths': src_seq_lengths,
            'src_seq': src_seq_tensor,
            'src_mask': src_mask,
            'src_extend_seq': src_extend_tensor,
            'oovs': oov_zeros,
            'oov_list': oovs,
            'inp_seq': inp_seq_tensor,
            'tgt_seq': tgt_seq_tensor,
            'src_recover': src_sent_seq_recover,
            'tgt_tree': tgt_trees,
            'att_label': smooth_one_hot,
            'con_seq': content_seq_tensor,
            'con_lengths': content_seq_lengths,
            'con_mask': content_mask,
            'con_recover': content_seq_recover
        }


def phrase_span(sent, parse_starts):
    parse_starts.append(len(sent))
    insert_eop = []
    for i in range(len(parse_starts) - 1):
        insert_eop += sent[parse_starts[i]:parse_starts[i + 1]] + ['<eop>']
    return insert_eop


def convert(args, tree, label, height):
    node = Node(label)
    if height == args.tree_height2:
        return (node, height)

    heights = [height]
    for child in tree[label]:
        if child in tree:
            kid, height_kid = convert(args, tree, child, height + 1)
            heights.append(height_kid)
            node.addkid(kid)
    return (node, max(heights))


# Removes leaves of tree
def remove_leaves_from_tree(tree, root, bpe=False, pos=False):
    mod_list = []
    for child in tree[root]:
        if child in tree:
            mod_list.append(child)
            remove_leaves_from_tree(tree, child, bpe, pos)
    if mod_list == [] and bpe:
        mod_list = [len(tree[root][0].split())]

    if mod_list == [] and pos:
        mod_list = [len(tree[root])]
    tree[root] = mod_list


def remove_leaves_from_trees(trees, bpe=False, pos=True):
    for tree in trees:
        remove_leaves_from_tree(tree, 'ROOT', bpe, pos)


def remove_pos_from_tree(tree, root, cnt, ht, pos_tags):
    words = []
    child_list = []
    for idx, child in enumerate(tree[root]):
        if child in tree:
            if child.rsplit('-', maxsplit=1)[0] in pos_tags:
                # if '<POS>' not in cnt:
                #     cnt['<POS>'] = 1
                words.append(tree[child][0])
                if ht > 2:
                    tree.pop(child)
            # child_list.append('<POS>'+'-'+str(cnt['<POS>']))
            else:
                child_list.append(child)

        # at first has pop operation .
        if child in tree:
            remove_pos_from_tree(tree, child, cnt, ht + 1, pos_tags)
    if ht > 2:
        if len(words) != 0:
            child_list.insert(0, '<POS>' + '-' + str(cnt['<POS>']))
            tree['<POS>' + '-' + str(cnt['<POS>'])] = words
            cnt['<POS>'] += 1
            tree[root] = child_list


def remove_pos_from_trees(trees, pos_tags):
    for tree in trees:
        remove_pos_from_tree(tree, 'ROOT', cnt={'<POS>': 1}, ht=1, pos_tags=pos_tags)


def trim_tree(tree, height, root, cur_height):
    if len(tree[root]) == 0:
        return [1]
    if type(tree[root][0]) == int:
        subword_len = tree[root][0]
        tree[root] = []
        return [subword_len]
    retlist = []
    if cur_height >= height:
        for child in tree[root]:
            retlist += trim_tree(tree, height, child, cur_height + 1)
            tree.pop(child)
        tree[root] = []
        return [np.sum(retlist)]
    else:
        for child in tree[root]:
            retlist += trim_tree(tree, height, child, cur_height + 1)
        return retlist


def trim_trees(trees, height):
    phrase_ends = []
    trimmed_trees = []
    for tree in trees:
        # tree = json.loads(tree)
        cur_phrase_lengths = trim_tree(tree, height, 'ROOT', 1)
        cur_phrase_ends = [0]
        for l in cur_phrase_lengths:
            cur_phrase_ends.append(cur_phrase_ends[-1] + l)
        cur_phrase_ends.pop()
        phrase_ends.append(cur_phrase_ends)
        trimmed_trees.append(tree)
    return trimmed_trees, phrase_ends


def prune_trees(args, pairs, ht, is_train=True):
    if args.dynamic_tree:
        # PROCESSING FOR DYNAMIC HEIGHT DECODING
        tgt_trees = []
        tgt_phrase_starts = []
        for i in range(len(pairs['tgt_tree'])):
            _, max_tgt_ht = convert(args, json.loads(pairs['tgt_tree'][i]), 'ROOT', 1)
            if is_train:
                if max_tgt_ht < 5:
                    ht = 4
                else:
                    ht = random.choice(range(5, max_tgt_ht + 1))
            else:
                ht = max_tgt_ht

            ptreest = trim_trees([pairs['tgt_tree'][i]], ht)
            tgt_tree, tgt_st = ptreest[0][0], ptreest[1][0]
            tgt_trees.append(tgt_tree)
            tgt_phrase_starts.append(tgt_st)

    else:
        tgt_trees, tgt_phrase_starts = trim_trees(pairs['tgt_tree'], ht)
    return tgt_trees, tgt_phrase_starts


def trees_travel(trees, root):
    paths = []
    for tree in trees:
        path = []
        deep_first_travel(path, tree, root)
        paths.append(path)
    return paths


def deep_first_travel(path, tree, root):
    path.append('(')
    path.append(root.rsplit('-', maxsplit=1)[0])
    for child in tree[root]:
        if len(child) != 0:
            deep_first_travel(path, tree, child)
    path.append(')')


def dfs(root, s):
    # label = root._label.split()[0].replace('-','')
    label = root.label()
    # label = re.findall(r'-+[A-Z]+-|[A-Z]+\$|[A-Z]+|\.', root._label)[0]
    tree_dict = {label: []}
    # print(leaf, root._label)
    # if len(root._label.split()) > 1:
    #     tree_dict[label].append(root._label.split()[1])

    for child in root:
        if type(child) is str:
            tree_dict[label] = child
            return tree_dict
        # if child._label.split()[]
        else:
            tree_dict[label].append(dfs(child, s))
    return tree_dict


def dfsopti(tree, root, tree_dict, label, cnt):
    tree_dict[label] = []

    if type(tree[root]) is str:
        tree_dict[label].append(tree[root])
    else:
        for child in tree[root]:
            if type(child) is dict:
                child_label = list(child.keys())[0]
                if child_label not in cnt:
                    cnt[child_label] = 0
                cnt[child_label] += 1
                child_label = child_label + '-' + str(cnt[child_label])
                tree_dict[label].append(child_label)
                dfsopti(child, list(child.keys())[0], tree_dict, child_label, cnt)


def tree_to_dict(parse):
    tree = Tree.fromstring(parse)
    p_dict = dfs(tree, '')
    tree = p_dict
    tree_dict = {}
    cnt = {'ROOT': 0}
    dfsopti(tree, 'ROOT', tree_dict, 'ROOT', cnt)
    return tree_dict


def trees_to_dict(parses):
    tree_dicts = []
    for parse in parses:
        tree_dicts.append(tree_to_dict(parse))
    return tree_dicts
