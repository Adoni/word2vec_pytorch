import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    def __init__(self, emb_size, emb_dimension):
        '''
        emb_size: the count of nodes which have embedding
        emb_dimension: embedding dimention
        '''
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_u, neg_v):
        # print(
        #     self.u_embeddings(Variable(torch.LongTensor([0]))).data.numpy()[0]
        #     [:10])
        losses = []
        emb_u = self.u_embeddings(Variable(torch.LongTensor(pos_u)))
        emb_v = self.v_embeddings(Variable(torch.LongTensor(pos_v)))
        score = torch.dot(emb_u, emb_v)
        score = F.logsigmoid(score)
        losses.append(-1 * score)
        neg_emb_u = self.u_embeddings(Variable(torch.LongTensor(neg_u)))
        neg_emb_v = self.v_embeddings(Variable(torch.LongTensor(neg_v)))
        neg_score = torch.dot(neg_emb_u, neg_emb_v)
        neg_score = F.logsigmoid(neg_score)
        losses.append(-1 * neg_score)
        return sum(losses)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.data.numpy()
        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))


def test():
    model = SkipGramModel(100, 100)
    id2word = dict()
    for i in range(100):
        id2word[i] = str(i)
    model.save_embedding(id2word)


if __name__ == '__main__':
    test()
