import numpy as np 
from keras.preprocessing.text import one_hot
from keras_preprocessing import sequence
from keras import Sequential
from keras.models import load_model
from keras.layers import Embedding, LSTM, Dense
import matplotlib.pyplot as plt 
import sys

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

x_train = encoded_positive[:7000] + encoded_negative[:7000]
x_test = encoded_positive[7000:] + encoded_negative[7000:]
y_train = [1] * 7000 + [0] * 7000
y_test = [1] * 2926 + [0] * 2704

max_review_length = 350
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)


if len(sys.argv) <= 1:
    ### Build the model ###
    embedding_vector_length = 32
    model = Sequential()
    model.add(Embedding(vocab, embedding_vector_length, input_length=max_review_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    ### Model training ###
    model.fit(x_train, y_train, validation_split=0.15, nb_epoch=3, batch_size=64)

    ### Model evaluation ###
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    model.save('model.h5')
    del model

    '''initial_accuracy = scores[1]*100

    vector_lengths = [16, 64, 128, 256]
    lstm_sizes = [25, 50, 150, 250]

    vector_acc = []
    lstm_acc = []

    for length in vector_lengths:
        ### Build the model ###
        embedding_vector_length = length
        model = Sequential()
        model.add(Embedding(vocab, embedding_vector_length, input_length=max_review_length))
        model.add(LSTM(100))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        ### Model training ###
        model.fit(x_train, y_train, validation_split=0.15, nb_epoch=3, batch_size=64)

        ### Model evaluation ###
        scores = model.evaluate(x_test, y_test, verbose=0)
        print("Accuracy for vector length ", length, " : %.2f%%" % (scores[1]*100))
        vector_acc.append(scores[1]*100)

    vector_acc = vector_acc[:1] + [initial_accuracy] + vector_acc[1:]
    vector_lengths = vector_lengths[:1] + [32] + vector_lengths[1:]

    plt.plot(vector_lengths, vector_acc)
    plt.show()

    for size in lstm_sizes:
        ### Build the model ###
        embedding_vector_length = 32
        model = Sequential()
        model.add(Embedding(vocab, embedding_vector_length, input_length=max_review_length))
        model.add(LSTM(size))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        ### Model training ###
        model.fit(x_train, y_train, validation_split=0.15, nb_epoch=3, batch_size=64)

        ### Model evaluation ###
        scores = model.evaluate(x_test, y_test, verbose=0)
        print("Accuracy for lstm size ", size, ": %.2f%%" % (scores[1]*100))
        lstm_acc.append(scores[1]*100)

    lstm_acc = lstm_acc[:2] + [initial_accuracy] + lstm_acc[2:]
    lstm_sizes = lstm_sizes[:2] + [100] + lstm_sizes[2:]

    plt.plot(lstm_sizes, lstm_acc)
    plt.show()'''
else:
    ### Load and evaluate previous model ###
    model = load_model('model.h5')
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    ### Predict sentiment from reviews ###
    bad = "The app is constantly freezing"
    good = "This game is really fun and addictive"
    bad_ = one_hot(bad, negative_vocab)
    good_ = one_hot(good, positive_vocab)
    bad_encoded = sequence.pad_sequences([bad_], maxlen=max_review_length)
    good_encoded = sequence.pad_sequences([good_], maxlen=max_review_length)
    print(bad, "Sentiment: ", model.predict(np.array([bad_encoded][0]))[0][0])
    print(good, "Sentiment: ", model.predict(np.array([good_encoded][0]))[0][0])