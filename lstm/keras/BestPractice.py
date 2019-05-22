#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : BestPractice.py
# @Author: wu gang
# @Date  : 2019/5/22
# @Desc  : 
# @Contact: 752820344@qq.com


from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Embedding
from keras import datasets
from keras.preprocessing import sequence


def plot_acc_and_loss(history):
    from matplotlib import pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validtion acc')
    plt.legend()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


# 读取数据
max_features = 10000  # 我们只考虑最常用的10k词汇
maxlen = 500  # 每个评论我们只考虑100个单词
(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=max_features)
print(len(x_train), len(x_test))

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)  # 长了就截断，短了就补0

# 1. Stack SimpleRNN
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(64, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(16))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc']
)

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

plot_acc_and_loss(history)

# 2. Dropout for RNN
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(64, dropout=0.1, recurrent_constraint=0.5, return_sequences=True))
model.add(SimpleRNN(32, dropout=0.1, recurrent_constraint=0.5, return_sequences=True))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc']
)

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

plot_acc_and_loss(history)
