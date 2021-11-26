import collections
import math
# filename = '../data/super-para50w/train/src_sent.txt'
#
# sents = load_file_sent_or_parser(filename)
# percentage = 0.5


def word_document(sents):
    vocab = dict()
    for word_list in sents:
        for w in set(word_list):
            if w in vocab:
                vocab[w] += 1
            else:
                vocab[w] = 1
    return vocab


def content_word(sent, docs_n, vocab, tgt, percentage):
    # sent : word list .
    # vocab : word document occurrence .
    tf_counter = collections.Counter()
    tf_counter.update(sent)
    word_tfidf_idx = []
    length = len(sent)

    for idx, w in enumerate(sent):
        if w in ['<s>', '</s>']:
            continue

        try:
            word_occurence = vocab[w]
        except KeyError:
            word_occurence = 1
        tfidf = tf_counter[w] / length * math.log(docs_n / (1 + word_occurence))
        word_tfidf_idx.append((w, tfidf, idx))

    top_n = max(1, int(length * percentage))
    content_word = sorted(word_tfidf_idx, key=lambda x: x[1], reverse=True)[:top_n]
    content_word = sorted(content_word, key=lambda x: x[2])
    if tgt:
        content_word_list = ['<PAD>'] * (length+1) # start or end mark.
        for t in content_word:
            idx = t[2]
            c_word = t[0]
            content_word_list[idx]=c_word
    else:
        content_word_list, _, _ = list(zip(*content_word))
    return content_word_list


def select_content_words(sents, word_docs_occurence, document_n, tgt=False, ratio=0.4):
    corpus_content = []
    # word_docs_occurence = word_document(sents)
    for sent in sents:
        corpus_content.append(list(content_word(sent, document_n, word_docs_occurence, tgt=tgt, percentage=ratio)))
    return corpus_content

# word_occurence, document_num = pickle.load(open(path, 'rb'))
# contents = select_content_words(sents, word_occurence, document_num, ratio=0.4)


# lens = []
# for content in contents:
#     lens.append(len(content))
#
# print(min(lens))
# print(max(lens))
# print(lens.index(max(lens)))
