'''for i in range(len(positive)):
    if i >= len(positive):
        break
    if len(positive[i]) == 0:
        positive.pop(i)

for i in range(len(positive)):
    if i >= len(positive):
        break
    if len(positive[i]) == 0:
        print(i)

with open('negative10kmod.txt', 'w+') as f:
    for line in negative:
        f.write('%s\n' % line)

with open('positive10kmod.txt', 'w+') as f:
    for line in positive:
        f.write('%s\n' % line)

for i in range(len(negative)):
    if i >= len(negative):
        break
    if len(negative[i]) == 0:
        negative.pop(i)

for i in range(len(negative)):
    if i >= len(negative):
        break
    if len(negative[i]) == 0:
        print(i)'''

from nltk.tokenize import word_tokenize
import nltk
import re
from gensim.models import Word2Vec, KeyedVectors



negative = open('negative10kmod.txt','r').readlines()
negative = list(map(lambda s: s.strip(), negative))
negative = list(map(lambda s: re.sub('[^a-zA-z0-9\s]','',s), negative))



positive = open('positive10kmod.txt','r').readlines()
positive = list(map(lambda s: s.strip(), positive))
positive = list(map(lambda s: re.sub('[^a-zA-z0-9\s]','',s), positive))



negative_words = []
for line in negative:
    tok = word_tokenize(line)
    negative_words.append(tok) 

positive_words = []
for line in positive:
    tok = word_tokenize(line)
    positive_words.append(tok)

model = Word2Vec(positive_words, size=100)
vector = model.wv
vector = vector.get_keras_embedding(train_embeddings=True)
print(type(vector))
