from nltk.tokenize import word_tokenize

negative = open('negative10kmodified.txt','r').readlines()
negative = list(map(lambda s: s.strip(), negative))

positive = open('positive10kmodified.txt','r').readlines()
positive = list(map(lambda s: s.strip(), positive))

negative_words = []
for line in negative:
    tok = word_tokenize(line)
    negative_words.append(tok) 

positive_words = []
for line in positive:
    tok = word_tokenize(line)
    positive_words.append(tok)

print(negative_words)  