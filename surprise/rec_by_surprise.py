#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : rec_by_surprise.py
# @Author: wu gang
# @Date  : 2019/5/13
# @Desc  : 基于Surprise框架的推荐系统Demo
# @Contact: 752820344@qq.com
from surprise import Dataset
from surprise import KNNBasic
from surprise import evaluate, print_perf


def knn(data):
    data.split(n_folds=3)
    # We'll use the famous KNNBasic algorithm.
    knn = KNNBasic()
    # Evaluate performances of our algorithm on the dataset.
    perf = evaluate(knn, data, measures=['RMSE', 'MAE'])
    print_perf(perf)


def svd(data):
    from surprise.model_selection import GridSearchCV
    from surprise import SVD

    # 所需参数候选值：epoch、学习率和正则化参数
    # 共组合2*2*2=8种参数组合
    param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
                  'reg_all': [0.4, 0.6]}
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)

    # best RMSE score
    print(gs.best_score['rmse'])
    # combination of parameters that gave the best RMSE score
    print(gs.best_params['rmse'])

    # 将各个参数组合的结果列表展示
    import pandas as pd
    results_df = pd.DataFrame.from_dict(gs.cv_results)
    print(results_df)


def read_item_names():
    """
    将电影名字和ID进行映射
    :return:
    """
    import io

    file_name = ('/Users/wugang/datasets/movie/ml-100k/u.item')
    rid_to_name = {}
    name_to_rid = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]
    return rid_to_name, name_to_rid


if __name__ == '__main__':
    # http://surprise.readthedocs.io/en/stable/index.html
    # http://files.grouplens.org/datasets/movielens/ml-100k-README.txt

    # Load the movielens-100k dataset (download it if needed),
    # and split it into 3 folds for cross-validation.
    data = Dataset.load_builtin('ml-100k')
    # knn(data)
    # svd(data)

    # # test read_item_names
    # rid_to_name, name_to_rid = read_item_names()
    # toy_story_raw_id = name_to_rid['Now and Then (1995)']
    # print(toy_story_raw_id)
    # raw_id_toy_story = rid_to_name['1053']
    # print(raw_id_toy_story)

    train_data = data.build_full_trainset()
    # 皮尔逊系数计算相似度，基于物品的推荐
    sim_options = {'name': 'pearson_baseline', 'user_based': False}
    from surprise import KNNBaseline

    knn = KNNBaseline(sim_options=sim_options)
    knn.fit(train_data)

    rid_to_name, name_to_rid = read_item_names()
    toy_story_raw_id = name_to_rid['Now and Then (1995)']
    print("数据中的ID：%s" % toy_story_raw_id)

    # toy_story_inner_id在实际要计算的矩阵中的ID。
    toy_story_inner_id = knn.trainset.to_inner_iid(toy_story_raw_id)
    print("矩阵中的ID：%s" % toy_story_inner_id)
    toy_story_neighbors = knn.get_neighbors(toy_story_inner_id, k=10)
    print("最近邻：%s" % toy_story_neighbors)

    # 对近邻集合中对ID进行转换为名字
    toy_story_neighbors = (knn.trainset.to_raw_iid(inner_id)
                           for inner_id in toy_story_neighbors)
    toy_story_neighbors = (rid_to_name[rid]
                           for rid in toy_story_neighbors)
    print()
    print('推荐的Top 10 :')
    for movie in toy_story_neighbors:
        print(movie)
