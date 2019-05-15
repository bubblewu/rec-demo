#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : douban.py
# @Author: wu gang
# @Date  : 2019/5/14
# @Desc  : 爬取豆瓣上的电影数据
# @Contact: 752820344@qq.com
import time

import logging
import pandas as pd
import random
import requests


class DouBanSpider(object):
    """
    豆瓣爬虫
    """

    def __init__(self, tag, movie_type, country, genres):
        """
        :param tag: 影视形式
        :param movie_type: 电影类型
        :param country: 国家
        :param genres: 电影特色
        """
        # 抓包获取数据接口：其中sort表示排序方式,需与传参保持一致（U近期热门（默认）、T标记最多、S评分最高、R最新上映）
        self.base_url = 'https://movie.douban.com/j/new_search_subjects?sort=S&range=0,10&'
        # self.full_url = self.base_url + '{query_params}'
        # 从User-Agents中选择一个User-Agent
        # self.headers = {'User-Agent':random.choice(User_Agents)}
        self.headers = {'User-Agent': 'Mozilla/4.0'}
        # self.proxies = {'http':random.choice(Agent_IP)}
        # 可选参数
        self.form_tag = tag  # 影视形式
        self.type_tag = movie_type  # 类型
        self.countries_tag = country  # 地区
        self.genres_tag = genres  # 特色
        # 默认参数
        self.sort = 'U'  # 排序方式,U近期热门（默认）、T标记最多、S评分最高、R最新上映
        self.range = 0, 10  # 评分范围

    def encode_query_params(self):
        """对请求参数对值进行编码处理"""
        all_tags = self.form_tag + ',' + self.genres_tag
        query_params = 'tags=' + all_tags + '&genres=' + self.type_tag + '&countries=' + self.countries_tag
        self.full_url = self.base_url + query_params + '&start='

    def download_movies(self, offset):
        """下载电影信息
        :param offset: 控制一次请求的影视数量
        :return resp:请求得到的响应体"""
        full_url = self.full_url + str(offset)
        print(full_url)
        resp = None
        try:
            # 方法1.USER_AGENT配置,仿造浏览器访问 headers
            # 方法2.伪造Cookie，解封豆瓣IP ,cookies = jar
            # jar = requests.cookies.RequestsCookieJar()
            # jar.set('bid', 'ehjk9OLdwha', domain='.douban.com', path='/')
            # jar.set('11', '25678', domain='.douban.com', path='/')
            # 方法3.使用代理IP proxies
            resp = requests.get(full_url, headers=self.headers)  # ,proxies=self.proxies
        except Exception as e:
            print(resp)
            logging.error(e)
        return resp

    def get_movies(self, resp):
        """获取电影信息
        :param resp: 响应体
        :return movies:爬取到的电影信息"""
        if resp:
            if resp.status_code == 200:
                # 获取响应文件中的电影数据
                movies = dict(resp.json()).get('data')
                if movies:
                    # 获取到电影了,
                    return movies
                else:
                    # 响应结果中没有电影了!
                    # print('已超出范围!')
                    return None
            else:
                # 关机
                print("poweroff")
                # os.system("poweroff")
        else:
            # 没有获取到电影信息
            return None

    def save_movies(self, movies):
        """把请求到的电影保存到csv文件中
        :param movies:提取到的电影信息
        """
        # 判断爬取的网页是否为空
        if len(str(movies)) < 20:
            return False

        data_list = []
        # 输出格式：类型、主演、地区、导演、特色、评分、电影名、海报、地址
        names = ['类型', '主演', '地区', '导演', '特色', '评分', '电影名', '海报', '地址']
        for movie in movies:
            item = []
            # 类型
            item.append(self.type_tag)
            # 主演
            casts_list = movie['casts']
            casts = ''
            casts_list_count = len(casts_list)
            for i in range(casts_list_count):
                casts = casts + casts_list[i]
                if i != casts_list_count - 1:
                    casts = casts + '|'
            item.append(casts)
            # 地区
            item.append(self.countries_tag)
            # 导演
            directors_list = movie['directors']
            directors_list_count = len(directors_list)
            directors = ''
            for i in range(directors_list_count):
                directors = directors + directors_list[i]
                if i != directors_list_count - 1:
                    directors = directors + '|'
            item.append(directors)
            # 特色
            item.append(self.genres_tag)
            # 评分
            item.append(movie['rate'])
            # 电影名
            item.append(movie['title'])
            # 海报
            item.append(movie['cover'])
            # 地址
            item.append(movie['url'])

            data_list.append(item)

        frame = pd.DataFrame(data_list, columns=names)
        # 不保留索引，不保留标题，追加写入
        frame.to_csv('../../data/douban/movie.csv', index=0, header=0, mode='a')
        # frame.to_csv('../../data/douban/movie.csv', index=0, header=True, mode='a')

        return True


def get_tags():
    # https://movie.douban.com/tag/#/
    # 全部形式
    all_tags = ['电影', '电视剧', '综艺', '动漫', '纪录片', '短片']
    # 全部类型
    type_tags = ['剧情', '喜剧', '动作', '爱情', '科幻', '动画', '悬疑', '惊悚', '恐怖', '犯罪', '同性', '音乐', '歌舞', '传记', '历史',
                 '战争', '西部', '奇幻', '冒险', '灾难', '武侠', '情色']
    # 全部地区
    country_tags = ['中国大陆', '美国', '香港', '台湾', '日本', '韩国', '英国', '法国', '德国', '意大利', '西班牙', '印度',
                    '泰国', '俄罗斯', '伊朗', '加拿大', '澳大利亚', '爱尔兰', '瑞典', '巴西', '丹麦']
    # 全部年代
    # 全部特色
    genres_tags = ['经典', '青春', '文艺', '搞笑', '励志', '魔幻', '感人', '女性', '黑帮']
    return all_tags, type_tags, country_tags, genres_tags


def main():
    all_tags, type_tags, country_tags, genres_tags = get_tags()
    for tag in all_tags:
        if tag != '电影':
            continue
        for movie_type in type_tags:
            # if movie_type != '战争':
            #     continue
            for country in country_tags:
                for genres in genres_tags:
                    print("开始抓取 [%s-%s-%s-%s] " % (tag, movie_type, country, genres))
                    # 1. 初始化工作,设置请求头等
                    spider = DouBanSpider(tag=tag, movie_type=movie_type, country=country, genres=genres)
                    # 2. 对信息进行编码处理,组合成有效的URL组合成有效的URL
                    spider.encode_query_params()

                    offset = 0
                    flag = True
                    while flag:
                        # 3. 下载影视信息
                        reps = spider.download_movies(offset)
                        print(reps)
                        # 4.提取下载的信息
                        movies = spider.get_movies(reps)
                        # 5. 保存数据到csv文件
                        flag = spider.save_movies(movies)
                        print(offset, flag)
                        offset += 30
                        # 控制访问速度(访问太快会被封IP)
                        time.sleep(random.randint(4, 8))
            time.sleep(random.randint(20, 30))
        time.sleep(random.randint(40, 50))


if __name__ == '__main__':
    main()
