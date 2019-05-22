#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : sin_predict.py
# @Author: wu gang
# @Date  : 2019/5/22
# @Desc  : 预测正弦函数sin
# @Contact: 752820344@qq.com

"""
因为循环神经网 络模型预测的是离散时刻的取值，所以在程序中需要将连续的sin函数曲线离散化。
所谓`离散化`就是在一个给定的区间[O, MAX]内，通过有限个采样点模拟一个连续的曲线。
比如 在以下程序中每隔SAMPLEITERVAL对sin函数进行一次来样，来样得到的序列就是sin 函数离散化之后的结果。
以下程序为预测离散化之后的 sin 函数。
"""

import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt


def generate_data(seq, timesteps=10):
    X = []
    y = []
    # 输入数据是10个时间步的，去预测这10个时间步的后面一个的数据,
    # 即用sin函数前面的time_step个点的信息，去预测第i + time_step的函数值
    for i in range(len(seq) - timesteps):
        X.append([seq[i: i + timesteps]])
        y.append([seq[i + timesteps]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def lstm_model(X, y, is_training, hidden_size=30, num_layers=2):
    # 使用多层的LSTM结构
    cell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.BasicLSTMCell(hidden_size) for _ in range(num_layers)]
    )
    # 使用TensorFlow接口将多层的LSTM结构连接成RNN网络，并计算其前向传播结果
    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    # outputs是顶层LSTM在每一步的输出结果，它的维度是[batch_size, time, hidden_size]
    # 在这个问题中，我们只关注最后一个时刻的输出结果
    output = outputs[:, -1, :]

    # 对LSTM网络的输出再做加一层全连接层并计算损失。这里默认的损失为平均
    # 平方差损失函数
    # 损失函数（loss function）是用来估量你模型的预测值f(x)与真实值Y的不一致程度，它是一个非负实值函数,
    # 通常使用L(Y, f(x))来表示，损失函数越小，模型的鲁棒性就越好。
    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)
    # 只在训练时计算损失函数和优化步骤。测试时直接返回预测结果
    if not is_training:
        return predictions, None, None

    # 计算损失函数
    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)
    # 创建模型优化器，并得到优化步骤
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.train.get_global_step(),
        optimizer='Adagrad',
        learning_rate=0.1
    )
    return predictions, loss, train_op


def train(sess, train_X, train_y, batch_size=32, training_steps=10000, hidden_size=30, num_layers=2):
    # 将训练数据以数据集的方式提供给计算图
    # tf.data.Dataset.from_tensor_slices函数接收一个array并返回一个表示array切片的tf.data.Dataset。
    ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    # repeat方法在读取到组后的数据时重启数据集。要限制epochs的数量，可以设置count参数。
    # shuffle方法使用一个固定大小的缓冲区来随机对数据进行shuffle。设置大于数据集中sample数目的buffer_size可以确保数据完全混洗。
    # batch方法累计样本并堆叠它们，从而创建批次。
    ds = ds.repeat().shuffle(1000).batch(batch_size)
    # Build the Iterator, and return the read end of the pipeline.
    X, y = ds.make_one_shot_iterator().get_next()

    # 调用模型，得到预测结果、损失函数和训练操作
    # tf.variable_scope("model"): 返回一个用于定义创建variable（层）的op的上下文管理器。model为要打开的范围。
    with tf.variable_scope("model"):
        predictions, loss, train_op = lstm_model(X, y, True, hidden_size=hidden_size, num_layers=num_layers)

    # 初始化模型的参数
    sess.run(tf.global_variables_initializer())
    for i in range(training_steps):
        _, l = sess.run([train_op, loss])
        if i % 100 == 0:
            print("train step: " + str(i) + ", loss: " + str(l))


def run_eval(sess, test_X, test_y, testing_count=1000):
    # 将测试数据以数据集的方式提供给计算图
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()
    # 调用模型得到计算结果。这里不需要输入真实的y值
    with tf.variable_scope("model", reuse=True):
        prediction, _, _ = lstm_model(X, [0.0], False)
    # 预测值集合
    predictions = []
    # 真实值集合
    labels = []
    for i in range(testing_count):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)

    # 计算rmse作为评价指标
    # 通过np.squeeze()函数转换后，要显示的数组变成了秩为1的数组，即（10，）
    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    # RMSE均方根误差亦称标准误差：
    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    print("Mean Square Error is: %f" % rmse)

    # 对预测对sin函数曲线进行绘制图像
    plt.figure()
    plt.plot(predictions, label='predictions')
    plt.plot(labels, label='real_sin')
    plt.legend()
    plt.show()


def main():
    # LSTM中隐藏节点的个数
    HIDDEN_SIZE = 30
    # LSTM的层数
    NUM_LAYERS = 2

    # 循环神经网络的训练序列长度
    TIMESTEPS = 10
    # 训练轮数
    TRANING_STEPS = 10000
    # batch大小
    BATCH_SIZE = 32

    # 训练数据个数
    TRANING_EXAMPLES = 10000
    # 测试数据个数
    TESTING_EXAMPLES = 1000
    # 采样间隔
    SAMPLE_GAP = 0.01

    # 用正弦函数生成训练和测试数据集合。
    # numpy.linspace函数可以创建一个等差序列的数组，它常用的参数有三个参数，
    # 第一个参数表示起始值，第二个参数表示终止值，第三个参数表示数列的长度。
    # 例如：np.linspace(1，10, 10) 产生的数组是 arrray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])。
    test_start = (TRANING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
    test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP

    seq_train = np.linspace(0, test_start, TRANING_EXAMPLES + TIMESTEPS, dtype=np.float32)
    sin_seq_train = np.sin(seq_train)
    train_X, train_y = generate_data(sin_seq_train, timesteps=TIMESTEPS)
    seq_test = np.linspace(test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32)
    sin_seq_test = np.sin(seq_test)
    test_X, test_y = generate_data(sin_seq_test, timesteps=TIMESTEPS)

    with tf.Session() as sess:
        # 训练模型
        train(sess, train_X, train_y, batch_size=BATCH_SIZE, training_steps=TRANING_STEPS, hidden_size=HIDDEN_SIZE,
              num_layers=NUM_LAYERS)
        # 使用训练好的模型对测试数据进行预测
        run_eval(sess, test_X, test_y, testing_count=TESTING_EXAMPLES)


if __name__ == '__main__':
    main()
