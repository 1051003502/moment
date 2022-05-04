# -*- coding: utf-8 -*-

"""
Created on Mar. 11, 2019.
tensorflow implementation of the paper:
Dong-Kyu Chae et al. "CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks," In Proc. of ACM CIKM, 2018.
@author: Dong-Kyu Chae (kyu899@agape.hanyang.ac.kr)

IMPORTANT: make sure that (1) the user & item indices start from 0, and (2) the index should be continuous, without any empy index.
"""

import random
import operator
import numpy as np
import codecs
from collections import defaultdict
from operator import itemgetter
import collections
import torch
from torch.autograd import variable
import pymysql

'''从数据库加载训练数据
dbIP   (str)数据库IP
dbName (str)数据库名称
dbUser (str)数据库用户名
dbPassword (str)数据库密码
dataType   （int） 1电影 2图书 3音乐
'''
def loadDataFromDatabase(dbIP,dbName,dbUser,dbPassword,dataType):
    trainSet = defaultdict(list)  # 字典默认值是一个list    trainSet['key_new'] 是个list
    #max_u_id = -1
    max_i_id = -1
    try:
        db = pymysql.connect(host='121.36.33.158', port=3306, user='devuser', passwd='Bishechuji1.!', db='moment_dev_commonuser', charset='utf8')
        cursor = db.cursor()
        if(dataType=="movie"):
            sql = "SELECT * FROM `movie_rate` "
        elif(dataType=="book"):
            sql = "SELECT * FROM `book_rate` "
        elif(dataType=="music"):
            sql = "SELECT * FROM `music_rate` "
        cursor.execute(sql)
        data = cursor.fetchall()
        for userId, itemId, rating in data:
            userId = int(userId)
            itemId = int(itemId)

            # note that we regard all the observed ratings as implicit feedback
            trainSet[userId].append(itemId)

            #max_u_id = max(userId, max_u_id)
            max_i_id = max(itemId, max_i_id)
        cursor.close()
        db.close()
    except Exception as e:
        print("数据库连接异常")
        print(e.args)
        return 1
    return trainSet,len(trainSet),max_i_id+1


''' 返回量trainVector userMaskDict batchCount
'''
def to_Vectors(trainSet, userCount, itemCount):
    # assume that the default is itemBased

    userMaskDict = defaultdict(lambda: [0] * itemCount)
    batchCount = userCount  # 改动  直接写成userCount


    trainDict = defaultdict(lambda: [0] * itemCount)

    for userID, i_list in trainSet.items():
        for itemId in i_list:
            userMaskDict[userID][itemId] = -99999
            trainDict[userID][itemId] = 1.0

    trainVector = []
    for userID in trainSet.keys():
        trainVector.append(trainDict[userID])

    return (torch.Tensor(trainVector)), userMaskDict, batchCount


if __name__=="__main__":
    trainSet, userCount, itemCount=loadDataFromDatabase(0,0,0,0,0)
    trainVector, testMaskDict, batchCount = to_Vectors(trainSet, userCount, itemCount)
