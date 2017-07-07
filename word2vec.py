from input_data import InputData
import numpy
from model import SkipGramModel
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys


class Word2Vec:
    def __init__(self,
                 input_file,
                 emb_dimension=100,
                 batch_size=100,
                 window_size=5,
                 iteration=5):
        self.data = InputData(input_file)
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.window_size = window_size
        self.iteration = iteration
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)
        self.optimizer = optim.SGD(self.skip_gram_model.parameters(), lr=0.01)

    @profile
    def train(self):
        batch_count = self.iteration * self.data.sentence_count / self.batch_size
        for _ in tqdm(range(batch_count)):
            sentences = self.data.get_batch_sentences(self.batch_size)
            pos_word_pairs = []
            for sentence in sentences:
                for i, u in enumerate(sentence):
                    for j, v in enumerate(sentence[max(
                            i - self.window_size, 0):i + self.window_size]):
                        if i == j:
                            continue
                        pos_word_pairs.append((u, v))
            neg_word_pair = self.data.negative_sampling(pos_word_pairs, 5)
            pos_u = [pair[0] for pair in pos_word_pairs]
            pos_v = [pair[1] for pair in pos_word_pairs]
            neg_u = [pair[0] for pair in neg_word_pair]
            neg_v = [pair[1] for pair in neg_word_pair]
            self.optimizer.zero_grad()
            loss = self.skip_gram_model.forward(pos_u, pos_v, neg_u, neg_v)
            loss.backward()
            self.optimizer.step()


if __name__ == '__main__':
    w2v = Word2Vec(input_file=sys.argv[1])
    w2v.train()
