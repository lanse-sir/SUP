import numpy as np


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square


def load_embedding(fname, vocab, emb_dim=300, norm2one=False):
    embedding = dict()
    vocab_size = len(vocab)
    with open(fname, 'r', encoding='utf-8') as f_r:
        for line in f_r:
            if len(line.strip().split()) < 3:
                continue
            word, vec = line.strip().split(' ', 1)
            vec = np.fromstring(vec, sep=' ')
            embedding[word] = vec
    print('Total Pretrained word embedding is %d .' % (len(embedding)))
    # build pretrained embedding .
    scale = np.sqrt(3.0 / emb_dim)
    pretrained_emb = np.empty([vocab_size, emb_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, idx in vocab.items():
        if word in embedding:
            if norm2one:
                pretrained_emb[idx, :] = norm2one(embedding[word])
            else:
                pretrained_emb[idx, :] = embedding[word]
            perfect_match += 1
        elif word.lower() in embedding:
            word = word.lower()
            if norm2one:
                pretrained_emb[idx, :] = norm2one(embedding[word])
            else:
                pretrained_emb[idx, :] = embedding[word]
            case_match += 1
        else:
            pretrained_emb[idx, :] = np.random.uniform(-scale, scale, [1, emb_dim])
            not_match += 1
    pretrained_size = len(embedding)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
    pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / vocab_size))
    return pretrained_emb


if __name__ == '__main__':
    path = "F:/词向量/glove.42B.300d/glove.42B.300d.txt"
    load_embedding(path, '')
