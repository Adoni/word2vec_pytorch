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
                 input_file_name,
                 output_file_name,
                 emb_dimension=100,
                 batch_size=100,
                 window_size=5,
                 iteration=1):
        self.data = InputData(input_file_name)
        self.output_file_name = output_file_name
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.window_size = window_size
        self.iteration = iteration
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)
        self.optimizer = optim.SGD(self.skip_gram_model.parameters(), lr=0.025)

    # @profile
    def train(self):
        pair_count = self.data.sentence_length * (2 * self.window_size - 1) - (
            self.data.sentence_count - 1) * (1 + self.window_size
                                             ) * self.window_size
        print('Pair count: %d' % pair_count)
        batch_count = self.iteration * pair_count / self.batch_size
        process_bar = tqdm(range(batch_count))
        self.skip_gram_model.save_embedding(self.data.id2word,
                                            'begin_embedding.txt')
        for i in process_bar:
            pos_pairs = self.data.get_batch_pairs(self.batch_size)
            # pos_pairs, neg_pairs = self.data.get_pairs_by_neg_sampling(
            #     pos_pairs, 5)
            #
            pos_pairs, neg_pairs = self.data.get_pairs_by_huffman(pos_pairs)
            pos_u = [pair[0] for pair in pos_pairs]
            pos_v = [pair[1] for pair in pos_pairs]
            neg_u = [pair[0] for pair in neg_pairs]
            neg_v = [pair[1] for pair in neg_pairs]
            self.optimizer.zero_grad()
            loss = self.skip_gram_model.forward(pos_u, pos_v, neg_u, neg_v)
            loss.backward()
            self.optimizer.step()

            process_bar.set_description(
                "Loss: %0.5f, lr: %0.6f" %
                (loss.data[0], self.optimizer.param_groups[0]['lr']))
            if i * self.batch_size % 100000 == 0:
                lr = 0.025 * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        self.skip_gram_model.save_embedding(self.data.id2word,
                                            self.output_file_name)


if __name__ == '__main__':
    w2v = Word2Vec(input_file_name=sys.argv[1], output_file_name=sys.argv[2])
    w2v.train()
