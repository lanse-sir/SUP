import numpy as np
import torch
from torch.autograd import Variable


def add_noise(variable: Variable, pad_index: int, drop_probability: float = 0.1,
              shuffle_max_distance: int = 3) -> Variable:
    def perm(i):
        return i[0] + (shuffle_max_distance + 1) * np.random.random()

    new_variable = np.zeros((variable.size(0), variable.size(1)), dtype='int')
    variable = variable.data.cpu().numpy()
    for b in range(variable.shape[0]):
        sequence = variable[b]
        sequence = sequence[sequence != pad_index]
        sequence, reminder = sequence[:-1], sequence[-1:]
        if len(sequence) != 0:
            sequence = sequence[np.random.random_sample(len(sequence)) > drop_probability]
            sequence = [x for _, x in sorted(enumerate(sequence), key=perm)]
        sequence = np.concatenate((sequence, reminder), axis=0)
        sequence = list(np.pad(sequence, (0, variable.shape[1] - len(sequence)), 'constant',
                               constant_values=pad_index))
        new_variable[b, :] = sequence
    return Variable(torch.LongTensor(new_variable))


if __name__ == '__main__':
    input = torch.LongTensor([[1,2,3,4,5,6,0],[2,4,6,8,10,12,0]])
    new_input = add_noise(input,0)
    print(new_input)