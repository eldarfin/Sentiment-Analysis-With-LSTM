import numpy as np
from keras.preprocessing.text import one_hot
from keras_preprocessing import sequence
from keras import Sequential
from keras.models import load_model
from keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import sys
import json
import io
import numpy as np


def save_labels(data_dictionary):

    ratings = []
    for i in range(len(data_dictionary)):
        ratings.append(data_dictionary[i]['rating'])

    labels = np.array(ratings)

    # Assign 0 to negative and 1 to positive comments.
    # Decide according to the ratings (4 and 5 positive, below negative).
    labels[np.argwhere(labels <= 3)] = 0
    labels[np.argwhere(labels >= 4)] = 1
    labels = labels.astype(int)

    # Save labels as txt file.
    np.savetxt('labels.txt', labels.astype(int), fmt='%i')


def get_content(data_dictionary, data_type):
    """
    Extract content from dictionary.

    :param data_dictionary: dictionary
    :param data_type: str
    :return: list
    """

    content = []
    for i in range(len(data_dictionary)):
        content.append(data_dictionary[i][data_type])

    # Replace \n and \t with space.
    for i in range(len(content)):
        content[i] = content[i].replace('\n', ' ')
        content[i] = content[i].replace('\t', ' ')
        content[i] = content[i].lower()

    return content


#%% Dataset 2

with io.open('./data/json_data/all.json', 'r') as f:
    data_dict = json.load(f)

y2_test = np.loadtxt('./data/json_data/labels.txt').astype(int)

review_list = get_content(data_dict, 'comment')

vocab = len(sorted(set(review_list)))
encoded_vocab = [one_hot(line, vocab) for line in review_list]

max_review_length = 350
x2_test = sequence.pad_sequences(encoded_vocab, maxlen=max_review_length)

#%% Dataset 1

negative = open('./data/negative10kmod.txt', 'r').readlines()
positive = open('./data/positive10kmod.txt', 'r').readlines()

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


#%% Train classifier.

gg = 0
if gg <= 1:
    ### Build the model ###
    embedding_vector_length = 32
    dropout_rate = 0.2
    model = Sequential()
    model.add(Embedding(vocab, embedding_vector_length, input_length=max_review_length))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(100, dropout=0.5))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    ### Model training ###
    model.fit(x_train, y_train, validation_split=0.15, epochs=2, batch_size=64)

    ### Model evaluation ###
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    scores_2 = model.evaluate(x2_test, y2_test, verbose=0)
    print("Accuracy on dataset 2: %.2f%%" % (scores_2[1]*100))

    model.save('model.h5')
    model.save_weights('model_weights.h5')
    del model

else:
    ### Load and evaluate previous model ###
    model = load_model('model.h5')
    model.load_weights('model_weights.h5')
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

