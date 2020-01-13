import random
import math
import pickle

train = None
test = None
W = None
'''
splitData函数
简介：分割数据集为两部分：train,test
返回值：train,test  (已经分割好的)
0<=K<=M-1  M含义:均分为几份
'''
def splitData(data,M,K,seed):
    test=[]
    train=[]
    random.seed(seed)
    for user,item in data:
        if random.randint(0,M)==K:
            test.append([user,item])
        else:
            train.append([user,item])
    return train,test

'''
recall函数
简介：计算召回率
'''
def recall(train,test,N):
    hit=0
    all=0
    for user in train.keys():
        tu=test[user]
        rank=GetRecommendation(user,N)   #封装函数并不好 此处不应有GetRecommendation
        for item,pui in rank:
            if item in tu:
                hit+=1
        all+=len(tu)
    return hit / float(all)

'''
Precision函数
简介：计算准确率
'''
def Precision(train,test,K,N):
    hit=0
    all=0
    for user in test.keys():
        tu=test[user]
        tu=list(map(lambda x:x[0],tu))
        rank=GetRecommendation(user,K,N)
        for item,pui in rank:
            if item in tu:
                hit+=1
        all+=len(rank)
    return hit/all

'''
Coverage函数
简介：计算覆盖率
'''
def Coverage(train,test,N):
    recommend_items=set()
    all_items=set()
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)
        rank=GetRecommendation(user,N)
        for item,pui in rank:
            recommend_items.add(item)
    return len(recommend_items)/float(all_items)

'''
Popularity函数
简介：计算推荐结果的平均流行度 
'''
def Popularity(train,test,N):
    item_popularity=dict()
    for user,items in train.items():
        for item in items.keys():
            if item not in item_popularity:
                item_popularity[item]=0
            item_popularity[item]+=1
    ret=0
    n=0
    for user in train.keys():
        rank=GetRecommendation(user,N)
        for item,pui in rank:
            ret+=math.log(1+item_popularity[item])
            n+=1
    return ret/float(n)

'''
UserSimilarity函数
简介：计算余弦相似度
'''
def UserSimilarity(train):
    W=dict()
    for u in train.keys():
        if u not in W:
            W[u]=dict()
    for u in train.keys():
        for v in train.keys():
            if u==v:
                continue
            W[u][v]=len(train[u]&train[v])
            W[u][v]/=math.sqrt( len(train[u]) * len(train[v])*1.0 )
    return W

def UserSimilarity2(train):
    item_users=dict()
    for u,items in train.items():
        for i in items.keys():
            if i not in item_users:
                item_users[i]=set()
            item_users[i].add(u)

    C=dict()
    N=dict()
    for i,users in item_users.items():
        for u in users:
            N[u]+=1
            for v in users:
                if u==v:
                    continue
                C[u][v]+=1

    W=dict()
    for u in train.keys():
        if u not in W:
            W[u]=dict()
    for u,related_users in C.items():
        for v,cuv in related_users.items():
            W[u][v]=cuv/math.sqrt(N[u]*N[v])
    return W


'''
getData函数
简介：得到数据集
'''
def getData(filename):
    data = None
    with open(filename) as fp:
        data=[]
        for line in fp.readlines():
            line=line.strip("\n")
            oneitem=line.split("::")
            data.append(oneitem)
    return data
'''
getDataOnly2函数
简介：得到数据集 {UserID1:{MovieID1,MovieID2,...},UserID2:{...}, ...}
'''
def getDataOnly2(filename):
    data=getData(filename)
    for i in range(len(data)):
        data[i].pop()
        data[i][1]=(data[i][1],int(data[i].pop()))
        #data[i].pop()
    trainData,testData=splitData(data,8,1,0)
    train=dict()
    test=dict()
    for user,item in trainData:
        if user not in train:
            train[user] = set()
        train[user].add(item)
    for user,item in testData:
        if user not in test:
            test[user] = set()
        test[user].add(item)
    return train,test

'''
Recommend函数
简介：推荐
'''
def Recommend(user,train,W,K,N):
    rank=dict()
    interacted_items=list(map(lambda x:x[0],train[user]))
    for v,wuv in sorted(W[user].items(),key=itemgetter(1),reverse=True)[0:K]:
        for i,rvi in train[v]:
            if i in interacted_items:
                # we should filter items user interacted before
                continue
            if i not in rank:
                rank[i]=0
            rank[i]+=wuv*rvi

    return sorted(rank.items(),key=itemgetter(1),reverse=True)[0:N]
def itemgetter(location):
    def f(item):
        return item[location]
    return f
def getDataFromFile(filename):
    res=None
    with open(filename,"rb") as fp:
        res=pickle.load(fp)
    return res


def GetRecommendation(user,K,N):

    return Recommend(user,train,W,K,N)
if __name__ == '__main__':
    #train,test=getDataOnly2("../data/ml-1m/ratings.dat")
    #W=UserSimilarity(train)
    train = getDataFromFile("train_data")
    test = getDataFromFile("test_data")
    W = getDataFromFile("W_data")
    #rank1 = Recommend('1', train, W, 20, 10)
    precision1=Precision(train, test,5, 10)
    precision2 = Precision(train, test,10, 10)
    precision3 = Precision(train, test, 20,10)
    precision4 = Precision(train, test, 40, 10)
    precision5 = Precision(train, test, 80, 10)
    precision6 = Precision(train, test, 160, 10)
    print("K=5 N=10",precision1)#0.01
    print("K=10 N=10", precision2)  # 0.01
    print("K=20 N=10", precision3)  # 0.01
    print("K=40 N=10", precision4)  # 0.01
    print("K=80 N=10", precision5)  # 0.01
    print("K=160 N=10", precision6)  # 0.01

