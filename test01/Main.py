import function
if __name__ == '__main__':
    test_data=[['zhaoxin','买了手机'],['junfeng','2b'],['dog','bone'],['周鑫','guapi'],['周鹏飞','Game']\
                ,['panda','竹子']]
    train,test=function.splitData(test_data,2,0,3)
    print(train)
    print(test)
