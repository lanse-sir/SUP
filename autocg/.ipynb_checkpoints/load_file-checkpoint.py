import os


def list_to_file(filename, sents):
    with open(filename, 'w', encoding='utf-8') as f_w:
        for sent in sents:
            f_w.write(sent + '\n')


def load_sentence(file_name):
    sentences = []
    with open(file_name, 'r', encoding='utf-8') as f_r:
        for line in f_r:
            sentences.append(line.strip())
    return sentences


def load_train_data(path):
    sent_types = ['itis', 'there_be', 'active', 'passive']
    # itis: 0, there_be: 1, active: 2, passive: 3
    sents = []
    parser = []
    sent_type_idx = []
    for idx, type in enumerate(sent_types):
        fname = os.path.join(path, type + '.txt')
        parse_fname = os.path.join(path, type + '_parser.txt')
        with open(fname, 'r', encoding='utf-8') as f_r, open(parse_fname, 'r', encoding='utf-8') as f_p:
            for line in f_r:
                sents.append(line.strip().lower().split())
                sent_type_idx.append([idx])
            for line in f_p:
                parser.append(line.strip().lower().split())
    assert len(sents) == len(parser) == len(sent_type_idx)
    data = list(zip(sents, parser, sent_type_idx))
    return data


def load_test_data(path):
    control = os.path.join(path, 'control')
    input = os.path.join(path, 'input.txt')
    input_parse = os.path.join(path, 'input_parser.txt')
    refs = os.path.join(path, 'refs.txt')
    refs_parse = os.path.join(path, 'refs_parser.txt')
    sents = load_sentence(input)
    sents_parse = load_sentence(input_parse)
    refs_sent = load_sentence(refs)
    refs_sent_parse = load_sentence(refs_parse)
    control_type = [[int(c[0])] for c in load_sentence(control)]
    data = list(zip(sents, sents_parse, control_type, refs_sent, refs_sent_parse))
    return data


def mask_sent(sent, template):
    if type(sent) is str:
        sent = sent.split()
    if type(template) is str:
        template = template.split()
    mask_sent = []
    for word in sent:
        if word in template:
            continue
        else:
            mask_sent.append(word)
    return mask_sent


def load_data_pair(path, name='train'):
    sents = []
    templates = []
    refs = []
    # itis : 0, there_be: 1, act  ive: 2, passive: 3
    if name == 'train':
        sent_types = ['itis', 'there_be', 'passive']
        for idx, type in enumerate(sent_types):
            fname = os.path.join(path, name + '_' + type + '_input.txt')
            # parse_fname = os.path.join(path, type + '_parser.txt')
            with open(fname, 'r', encoding='utf-8') as f_r:
                for line in f_r:
                    sent, template = line.strip().lower().split('\t')
                    mask_sentence = mask_sent(sent, template)
                    sents.append(mask_sentence)
                    templates.append(template.split())
                    refs.append(sent.split())
    elif name == 'dev':
        sent_types = ['active_itis', 'active_passive', 'active_there_be']
        for idx, type in enumerate(sent_types):
            fname = os.path.join(path, name + '_' + type + '_input.txt')
            ref_fname = os.path.join(path, name + '_' + type + '_ref.txt')
            with open(fname, 'r', encoding='utf-8') as f_r, open(ref_fname, 'r', encoding='utf-8') as f_ref:
                for line in f_r:
                    sent, template = line.strip().lower().split('\t')
                    sents.append(sent.split())
                    templates.append(template.split())
                for line in f_ref:
                    refs.append(line.strip().lower().split())

    assert len(sents) == len(templates) == len(refs)
    data = list(zip(sents, templates, refs))
    return data


def save_to_file(sents, path):
    with open(path, 'w', encoding='utf-8') as f_w:
        for sent in sents:
            if type(sent) is str:
                f_w.write(sent + '\n')
            elif type(sent) is list:
                f_w.write(' '.join(sent) + '\n')
            elif type(sent) is int:
                f_w.write(str(sent) + '\n')
            else:
                line = ""
                for item in sent:
                    if type(item) is list:
                        line = line + ' '.join(item) + '\t'
                    elif type(item) is int:
                        line = line + str(item) + '\t'
                    else:
                        line = line + item + '\t'
                f_w.write(line + '\n')


def load_file_sent_or_parser(file_name, type='sent'):
    sentences = []
    with open(file_name, 'r', encoding='utf-8') as f_r:
        for line in f_r:
            if type == 'sent':
                sentences.append(line.strip().split())
            else:
                sentences.append(line.strip())
    return sentences


def load_template(filename):
    t = []
    with open(filename, 'r', encoding='utf-8') as f_r:
        for line in f_r:
            template, _ = line.strip().split(' ---- ')
            t.append(template.strip())
    return t


def load_vocab(fname):
    vocab = {}
    with open(fname, 'r', encoding='utf-8') as f_r:
        for line in f_r:
            word, idx = line.strip().split()
            vocab[word] = int(idx)
    return vocab


def load_pos_tag(fname):
    pos_tags = []
    with open(fname, 'r', encoding='utf-8') as f_r:
        for line in f_r:
            tag = line.strip().split('\t')
            pos_tags.append(tag[0])
    return pos_tags
