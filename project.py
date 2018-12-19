import numpy as np 
from keras.preprocessing.text import one_hot
from keras_preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense

### Preprocess data ###
negative = open('negative10kmod.txt','r').readlines()
positive = open('positive10kmod.txt','r').readlines()

temp = []
for line in negative:
    temp.append(line.strip())
negative = temp.copy()

temp = []
for line in positive:
    temp.append(line.strip())
positive = temp.copy()

positive = [line.lower() for line in positive]
negative = [line.lower() for line in negative]

### Create word vectors ###
negative_vocab = len(sorted(set(negative)))
positive_vocab = len(sorted(set(positive)))
vocab = len(sorted(set(positive+negative)))

encoded_negative = [one_hot(line, negative_vocab) for line in negative]
encoded_positive = [one_hot(line, positive_vocab) for line in positive]

x_train = encoded_positive[:6000] + encoded_negative[:6000]
x_validation = encoded_positive[6000:7000] + encoded_negative[6000:7000]
x_test = encoded_positive[7000:] + encoded_negative[7000:]
y_train = [1] * 6000 + [0] * 6000
y_validation = [1] * 1000 + [0] * 1000
y_test = [1] * 2926 + [0] * 2704

max_review_length = 350
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

### Build the model ###
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(vocab, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

### Model training ###
model.fit(x_train, y_train, validation_data=(x_test,  y_test), nb_epoch=3, batch_size=64)

### Model evaluation ###
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
