from numpy import random
from collections import deque
random.seed(12345)


class InputData:
    def __init__(self, fname):
        self.input_file_name = fname
        self.word_frequency = dict()
        self.input_file = open(self.input_file_name)
        self.sentence_length = 0
        self.sentence_count = 0
        for line in self.input_file:
            self.sentence_count += 1
            line = line.strip().split(' ')
            self.sentence_length += len(line)
            for w in line:
                try:
                    self.word_frequency[w] += 1
                except:
                    self.word_frequency[w] = 0
        self.word2id = dict()
        self.id2word = dict()
        wid = 0
        print('Sentence Length: %d' % (self.sentence_length))
        for w, c in self.word_frequency.items():
            if c < 5:
                self.sentence_length -= c
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            wid += 1
        print('Word Count: %d' % len(self.word2id))
        print('Sentence Length: %d' % (self.sentence_length))
        print('Sentence Length: %d' % sum(self.word_frequency.values()))
        self.word_pair_catch = deque()
        self.readed_sentence_count = 0

    def get_batch_sentences(self, batch_size):
        sentences = []
        for _ in range(batch_size):
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
            sentences.append(word_ids)
        return sentences

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
                        if i == j:
                            continue
                        self.word_pair_catch.append((u, v))
            # print('Catch: %d' % len(self.word_pair_catch))
        batch_pairs = []
        for _ in range(batch_size):
            batch_pairs.append(self.word_pair_catch.popleft())
        return batch_pairs

    # @profile
    def negative_sampling(self, pos_word_pair, count):
        neg_word_pair = []
        a = len(self.word2id) - 1
        for pair in pos_word_pair:
            i = 0
            neg_v = random.randint(low=0, high=a, size=count)
            neg_word_pair += zip([pair[0]] * count, neg_v)
        return neg_word_pair


def test():
    a = InputData('./zhihu.txt')


if __name__ == '__main__':
    test()
