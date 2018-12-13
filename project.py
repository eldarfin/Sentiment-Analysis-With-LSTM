from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from gensim.models import Word2Vec, KeyedVectors
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.layers.embeddings import Embedding
from keras.initializers import Constant



negative = open('negative10kmod.txt','r').readlines()
positive = open('positive10kmod.txt','r').readlines()


train_positive = positive[:7000]
test_positive = positive[7000:]
train_negative = negative[:7000]
test_negative = negative[7000:]

x_train = train_positive + train_negative
y_train = [1] * 7000 + [0] * 7000
x_test = test_positive + test_negative
y_test = [1] * 2926 + [0] * 2704

# Preprocess training set
temp = []
for line in x_train:
    tokens = word_tokenize(line)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [words for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    temp.append(words)

x_train = temp

# Preprocess test set
temp = []
for line in x_test:
    tokens = word_tokenize(line)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [words for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    temp.append(words)

x_test = temp

# Train word2vec model
model_train = Word2Vec(sentences=x_train, size=100, window=5, workers=4, min_count=1)
vector = model_train.wv
keras_embedding_layer = vector.get_keras_embedding(train_embeddings=True)

model = Sequential()
model.add(keras_embedding_layer)
model.add(LSTM(units=100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=25, validation_data=(x_test, y_test), verbose=2)
model.save('keras_lstm.model')


