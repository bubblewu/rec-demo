#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : latent_factor_model.py
# @Author: wu gang
# @Date  : 2019/5/13
# @Desc  : 基于TensorFlow的隐语义模型推荐
# @Contact: 752820344@qq.com

# 基于TensorFlow的隐语义模型推荐
# 数据：3900个电影；6040个用户

# Evaluate train times per epoch
import time
# Imports for data io operations
from collections import deque

import numpy as np
# Main imports for training
import tensorflow as tf
from six import next

import tensorflows.readers as readers


def get_data():
    """
    Reads file using the demiliter :: form the ratings file
    Columns are user ID, item ID, rating, and timestamp
    Sample data - 3::1196::4::978297539
    :return:
    """
    df = readers.read_file('/Users/wugang/datasets/movie/ml-1m/ratings.dat', sep="::")
    rows = len(df)
    # 基于位置选择的纯整数位置索引
    # 防止一个batch中都是一个用户，对数据进行乱序，也就是重排序
    # numpy.random,permutation(x)是返回一个被洗牌过的array
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    # Separate data into train and test, 90% for train and 10% for test
    split_index = int(rows * 0.9)
    # Use indices to separate the data
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)

    return df_train, df_test


def clip(x):
    """
    使数组或列表x中的值分布在1～5之间。
    np.clip截取函数, 将范围外的数强制转化为范围内的数.
    def clip(a, a_min, a_max, out=None): 将数组a中的所有数限定到范围a_min和a_max中，
    即az中所有比a_min小的数都会强制变为a_min，a中所有比a_max大的数都会强制变为a_max.
    :param x:
    :return:
    """
    return np.clip(x, 1.0, 5.0)


def model(user_batch, item_batch, user_num, item_num, dim=5, device="/cpu:0"):
    """

    :param user_batch:
    :param item_batch:
    :param user_num:
    :param item_num:
    :param dim: 矩阵的维度；
    :param device: 训练配置，默认为cpu执行
    :return:
    """
    with tf.device('/cpu:0'):
        # 指定命名域。防止在notebook中重复执行代码报错。
        # with tf.variable_scope('lsi', reuse=True):
        # Using a global bias term
        # 偏置项
        bias_global = tf.get_variable("bias_global", shape=[])
        # User and item bias variables
        # get_variable: Prefixes the name with the current variable scope
        # and performs reuse checks.
        w_bias_user = tf.get_variable("embd_bias_user", shape=[user_num])
        w_bias_item = tf.get_variable("embd_bias_item", shape=[item_num])
        # embedding_lookup: Looks up 'ids' in a list of embedding tensors
        # Bias embeddings for user and items, given a batch
        bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name="bias_user")
        bias_item = tf.nn.embedding_lookup(w_bias_item, item_batch, name="bias_item")
        # User and item weight variables
        # 权重（N*dim * dim*M = N*M），并使用标准差进行初始化
        w_user = tf.get_variable("embd_user", shape=[user_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        w_item = tf.get_variable("embd_item", shape=[item_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        # Weight embeddings for user and items, given a batch
        embd_user = tf.nn.embedding_lookup(w_user, user_batch, name="embedding_user")
        embd_item = tf.nn.embedding_lookup(w_item, item_batch, name="embedding_item")

    with tf.device(device):
        # infer 最终结果值
        # reduce_sum: Computes the sum of elements across dimensions of a tensor
        infer = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)
        infer = tf.add(infer, bias_global)
        infer = tf.add(infer, bias_user)
        infer = tf.add(infer, bias_item, name="svd_inference")
        # l2_loss: Computes half the L2 norm of a tensor without the sqrt
        # 正则化惩罚项矩阵
        regularizer = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item),
                             name="svd_regularizer")
    return infer, regularizer


def loss(infer, regularizer, rate_batch, learn_rate=0.01, reg=0.1, device="/cpu:0"):
    """

    :param infer: 预测最终结果值
    :param regularizer: 正则化惩罚项矩阵
    :param rate_batch: 真实值
    :param learn_rate: 梯度下降时的学习率
    :param reg: 正则化惩罚项，惩罚力度
    :param device: 运行环境配置
    :return:
    """
    with tf.device(device):
        # Use L2 loss to compute penalty
        # 预测值和真实值之间的差异
        cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
        penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="l2")
        cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))
        # 'Follow the Regularized Leader' optimizer
        # 梯度下降优化器
        train_op = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(cost)
    return cost, train_op


def train():
    # Constant seed for replicating training results
    # 指定随机种子（按一种随机方式执行），方便进行场景复现
    np.random.seed(42)

    # Number of users in the dataset
    u_num = 6040
    # Number of movies in the dataset
    i_num = 3952
    # Number of samples per batch
    # 迭代更新时的批次大小
    batch_size = 1000
    # Dimensions of the data, 15
    # 隐含因子的维度：6040*5 * 5*3952 = 6040*3952
    dims = 5
    # Number of times the network sees all the training data
    # 迭代次数
    max_epochs = 50

    # Device used for all computations
    place_device = "/cpu:0"

    # Read data from ratings file to build a TF model
    df_train, df_test = get_data()

    samples_per_batch = len(df_train) // batch_size
    print(samples_per_batch)
    print("Number of train samples %d, test samples %d, samples per batch %d" %
          (len(df_train), len(df_test), samples_per_batch))

    # Peeking at the top 5 user values
    print(df_train["user"].head())
    print(df_test["user"].head())
    # Peeking at the top 5 item values
    print(df_train["item"].head())
    print(df_test["item"].head())
    # Peeking at the top 5 rate values
    print(df_train["rate"].head())
    print(df_test["rate"].head())

    # Using a shuffle iterator to generate random batches, for training
    # 迭代训练
    # 利用ShuffleIterator取出指定的字段信息，并根据batch_size进行数据洗牌
    iter_train = readers.ShuffleIterator([df_train["user"],
                                          df_train["item"],
                                          df_train["rate"]],
                                         batch_size=batch_size)

    # Sequentially generate one-epoch batches, for testing
    # 指定一个epoch进行迭代测试
    iter_test = readers.OneEpochIterator([df_test["user"],
                                          df_test["item"],
                                          df_test["rate"]],
                                         batch_size=-1)

    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rate_batch = tf.placeholder(tf.float32, shape=[None])

    infer, regularizer = model(user_batch, item_batch, user_num=u_num, item_num=i_num, dim=dims, device=place_device)
    _, train_op = loss(infer, regularizer, rate_batch, learn_rate=0.0010, reg=0.05, device=place_device)

    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        print("%s\t%s\t%s\t%s" % ("Epoch", "Train Error", "Val Error", "Elapsed Time"))
        errors = deque(maxlen=samples_per_batch)
        start = time.time()
        for i in range(max_epochs * samples_per_batch):
            users, items, rates = next(iter_train)
            _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                                   item_batch: items, rate_batch: rates})
            pred_batch = clip(pred_batch)
            errors.append(np.power(pred_batch - rates, 2))
            if i % samples_per_batch == 0:
                train_err = np.sqrt(np.mean(errors))
                test_err2 = np.array([])
                for users, items, rates in iter_test:
                    pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                            item_batch: items})
                    pred_batch = clip(pred_batch)
                    test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
                end = time.time()

                print("%02d\t%.3f\t\t%.3f\t\t%.3f secs" % (
                    i // samples_per_batch, train_err, np.sqrt(np.mean(test_err2)), end - start))
                start = end

        saver.save(sess, './save/')


if __name__ == '__main__':
    train()
