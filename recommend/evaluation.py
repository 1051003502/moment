'''
衡量预测效果
groundTruth:真实结果
result:预测向量  此量经过了+testMaskVector操作处理  已经剔除掉训练集中用户有过反映的量
topN：取几个预测结果
'''
def computeTopNAccuracy(groundTruth,result,topN):
    result=result.tolist()
    for i in range(len(result)):
        result[i]=(result[i],i)
    result.sort(key=lambda x:x[0],reverse=True)
    #print(result)
    hit=0
    for i in range(topN):
        if(result[i][1] in groundTruth):hit=hit+1
    return hit
#computeTopNAccuracy(0,[1,2,5,9,100,-5,6,0],0)