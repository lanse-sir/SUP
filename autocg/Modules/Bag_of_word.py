import torch
import torch.nn as nn


class BOW(nn.Module):
    def __init__(self, in_size, output_size):
        super(BOW, self).__init__()
        self.input_size = in_size
        self.output_size = output_size

        self.out = nn.Linear(self.input_size, self.output_size)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, input, label):
        logits = self.out(input)
        scores = self.loss(logits, label)
        return scores
