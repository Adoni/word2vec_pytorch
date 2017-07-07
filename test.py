import sys
from sklearn.metrics.pairwise import cosine_similarity
import numpy
f=open(sys.argv[1])
f.readline()
all_embeddings=[]
all_words=[]
word2id=dict()
for i,line in enumerate(f):
    line=line.strip().split(' ')
    word=line[0]
    embedding=[float(x) for x in line[1:]]
    assert len(embedding)==100
    all_embeddings.append(embedding)
    all_words.append(word)
    word2id[word]=i
all_embeddings=numpy.array(all_embeddings)
while 1:
    word=input('Word: ')
    try:
        wid=word2id[word]
    except:
        print('Cannot find this word')
        continue
    embedding=all_embeddings[wid:wid+1]
    d = cosine_similarity(embedding, all_embeddings)[0]
    d=zip(all_words, d)
    d=sorted(d, key=lambda x:x[1], reverse=True)
    for w in d[:10]:
        if len(w[0])<2:
            continue
        print(w)
