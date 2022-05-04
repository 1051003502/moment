import torch
import torch.nn as nn
from torch.autograd import Variable
import model
import random
import evaluation
import numpy as np
import copy
class CFGAN:
    def __init__(self,trainSet,userCount,itemCount,trainVector,userMaskdict,batchCount,epochCount,pro_ZR,pro_PM,arfa):
        self.trainSet=trainSet
        self.userCount=userCount
        self.itemCount=itemCount
        self.trainVector=trainVector
        self.userMaskdict=userMaskdict
        self.batchCount=batchCount
        self.epochCount=epochCount
        self.pro_ZR=pro_ZR
        self.pro_PM=pro_PM
        self.arfa=arfa
        self.G_model=None
    def train(self):
        trainSet=self.trainSet
        userCount = self.userCount
        itemCount = self.itemCount
        trainVector = self.trainVector
        batchCount = self.batchCount
        epochCount = self.epochCount
        pro_ZR = self.pro_ZR
        pro_PM = self.pro_PM
        arfa= self.arfa
        G = model.generator(itemCount)
        D = model.discriminator(itemCount)
        criterion1 = nn.BCELoss()  # 二分类的交叉熵
        criterion2 = nn.MSELoss()
        d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)

        g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)

        G_step = 2
        D_step = 2
        batchSize_G = 32
        batchSize_D = 32
        realLabel_G = Variable(torch.ones(batchSize_G))
        realLabel_G = realLabel_G.view(batchSize_G, 1)
        # fakeLabel_G = Variable(torch.zeros(batchSize_G))
        realLabel_D = Variable(torch.ones(batchSize_D))
        realLabel_D = realLabel_D.view(batchSize_D, 1)
        fakeLabel_D = Variable(torch.zeros(batchSize_D))
        fakeLabel_D = fakeLabel_D.view(batchSize_D, 1)
        ZR = []
        PM = []
        for epoch in range(epochCount):  # 训练epochCount次
            print(epoch)
            if (epoch % 100 == 0):
                ZR = []
                PM = []
                for i in range(userCount):
                    ZR.append([])
                    PM.append([])
                    ZR[i].append(np.random.choice(itemCount, int(pro_ZR*itemCount), replace=False))
                    PM[i].append(np.random.choice(itemCount, int(pro_PM*itemCount), replace=False))
            for step in range(D_step):  # 训练D

                leftIndex = random.randint(1, userCount - batchSize_D - 1)
                realData = Variable(trainVector[leftIndex:leftIndex + batchSize_D])  # MD个数据成为待训练数据

                maskVector1 = Variable(trainVector[leftIndex:leftIndex + batchSize_D])
                for i in range(len(maskVector1)):
                    maskVector1[i][PM[leftIndex + i]] = 1
                Condition = realData  # 把用户反馈数据作为他的特征 后期还要加入用户年龄、性别等信息
                realData_result = D(realData, Condition)
                d_loss_real = criterion1(realData_result, realLabel_D)

                fakeData = G(realData)
                fakeData = fakeData * maskVector1
                fakeData_result = D(fakeData, realData)
                d_loss_fake = criterion1(fakeData_result, fakeLabel_D)

                d_loss = d_loss_real + d_loss_fake
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

            for step in range(G_step):  # 训练G
                # 调整maskVector2\3
                leftIndex = random.randint(1, userCount - batchSize_G - 1)
                realData = Variable(trainVector[leftIndex:leftIndex + batchSize_G])

                maskVector2 = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize_G]))
                maskVector3 = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize_G]))
                for i in range(len(maskVector2)):
                    maskVector2[i][PM[i + leftIndex]] = 1
                    maskVector3[i][ZR[i + leftIndex]] = 1
                fakeData = G(realData)
                g_loss2 = arfa * criterion2(fakeData, maskVector3)
                fakeData = fakeData * maskVector2
                g_fakeData_result = D(fakeData, realData)
                g_loss1 = criterion1(g_fakeData_result, realLabel_G)
                g_loss = g_loss1 + g_loss2

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
        self.G_model = G
        return 0
    def getMovieList(self,userID,length):
        if(self.G_model==None):
            return 2
        if(userID not in self.trainSet):return 1
        userIndex=list(self.trainSet.keys()).index(userID)
        result=self.G_model(self.trainVector[userIndex])
        result+=torch.Tensor(self.userMaskdict[userID])
        result = result.tolist()
        for i in range(len(result)):
            result[i] = (result[i], i)
        result.sort(key=lambda x: x[0], reverse=True)
        L=[]
        userIDList=list(self.trainSet.keys())
        for i in range(length):
            L.append(userIDList[result[i][1]])
        return L




