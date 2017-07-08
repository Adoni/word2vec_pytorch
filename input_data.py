import numpy
from collections import deque
from huffman import HuffmanTree
numpy.random.seed(12345)


class InputData:
    def __init__(self, fname):
        self.input_file_name = fname

        self.input_file = open(self.input_file_name)
        self.sentence_length = 0
        self.sentence_count = 0
        word_frequency = dict()
        for line in self.input_file:
            self.sentence_count += 1
            line = line.strip().split(' ')
            self.sentence_length += len(line)
            for w in line:
                try:
                    word_frequency[w] += 1
                except:
                    word_frequency[w] = 1
        self.word2id = dict()
        self.id2word = dict()
        wid = 0
        print('Sentence Length: %d' % (self.sentence_length))
        self.word_frequency = dict()
        for w, c in word_frequency.items():
            if c < 5:
                self.sentence_length -= c
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        self.word_count = len(self.word2id)
        print('Word Count: %d' % len(self.word2id))
        print('Sentence Length: %d' % (self.sentence_length))
        print('Sentence Length: %d' % sum(self.word_frequency.values()))
        self.word_pair_catch = deque()
        self.readed_sentence_count = 0
        self.init_sample_table()
        tree = HuffmanTree(self.word_frequency)
        self.huffman_positive, self.huffman_negative = tree.get_huffman_code_and_path(
        )
        print('---------')
        print(len(self.huffman_positive))

    def init_sample_table(self):
        self.sample_table = []
        sample_table_size = 1e8
        pow_frequency = numpy.array(self.word_frequency.values())**0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = numpy.round(ratio * sample_table_size)
        for wid, c in enumerate(count):
            self.sample_table += [wid] * int(c)
        self.sample_table = numpy.array(self.sample_table)

    # @profile
    def get_batch_pairs(self, batch_size):
        if len(self.word_pair_catch) < batch_size:
            # print('Readed Sentence Count: %d' % self.readed_sentence_count)
            self.readed_sentence_count += 10000
            for _ in range(10000):
                sentence = self.input_file.readline()
                if sentence is None or sentence == '':
                    self.input_file = open(self.input_file_name)
                    sentence = self.input_file.readline()
                word_ids = []
                for word in sentence.strip().split(' '):
                    try:
                        word_ids.append(self.word2id[word])
                    except:
                        continue
                for i, u in enumerate(word_ids):
                    for j, v in enumerate(word_ids[max(i - 5, 0):i + 5]):
                        assert u < self.word_count
                        assert v < self.word_count
                        if i == j:
                            continue
                        self.word_pair_catch.append((u, v))
            # print('Catch: %d' % len(self.word_pair_catch))
        batch_pairs = []
        for _ in range(batch_size):
            batch_pairs.append(self.word_pair_catch.popleft())
        return batch_pairs

    # @profile
    def get_pairs_by_neg_sampling(self, pos_word_pair, count):
        neg_word_pair = []
        a = len(self.word2id) - 1
        for pair in pos_word_pair:
            i = 0
            neg_v = numpy.random.choice(self.sample_table, size=count)
            neg_word_pair += zip([pair[0]] * count, neg_v)
        return pos_word_pair, neg_word_pair

    def get_pairs_by_huffman(self, pos_word_pair):
        neg_word_pair = []
        a = len(self.word2id) - 1
        for i in range(len(pos_word_pair)):
            pair = pos_word_pair[i]
            # print(pair[1])
            pos_word_pair += zip([pair[0]] *
                                 len(self.huffman_positive[pair[1]]),
                                 self.huffman_positive[pair[1]])
            neg_word_pair += zip([pair[0]] *
                                 len(self.huffman_negative[pair[1]]),
                                 self.huffman_negative[pair[1]])

        return pos_word_pair, neg_word_pair


def test():
    a = InputData('./zhihu.txt')


if __name__ == '__main__':
    test()
