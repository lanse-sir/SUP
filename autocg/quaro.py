import pandas
import benepar

filename = 'F:/NLPcorpus/quora-question-pairs/train.csv/train.csv'


def load_csv(filename):
    data = pandas.read_csv(filename)
    q1 = data['question1'].values.tolist()
    q2 = data['question2'].values.tolist()
    label = data['is_duplicate'].values.tolist()
    return q1, q2, label


def sent_pair_lable(q1, q2, lable, para=0):
    ori = []
    ref = []
    for s1, s2, l in zip(q1, q2, lable):
        if l == para:
            ori.append(s1)
            ref.append(s2)
    return ori, ref


def parsing(sentences):
    pass
