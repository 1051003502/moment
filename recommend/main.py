from flask import Flask
from flask import request
import pymysql
import data
import Algo

trainSet1, userCount1, itemCount1 = data.loadDataFromDatabase(0, 0, 0, 0, "movie")
trainVector1, userMaskDict1, batchCount1 = data.to_Vectors(trainSet1, userCount1, itemCount1)
CF1=Algo.CFGAN(trainSet1, userCount1, itemCount1, trainVector1, userMaskDict1, batchCount1, 10,0.5,0.7,0.03)
CF1.train()

trainSet2, userCount2, itemCount2 = data.loadDataFromDatabase(0, 0, 0, 0, "book")
trainVector2, userMaskDict2, batchCount2 = data.to_Vectors(trainSet2, userCount2, itemCount2)
CF2=Algo.CFGAN(trainSet2, userCount2, itemCount2, trainVector2, userMaskDict2, batchCount2, 10,0.5,0.7,0.03)
CF2.train()

trainSet2, userCount2, itemCount2 = data.loadDataFromDatabase(0, 0, 0, 0, "music")
trainVector2, userMaskDict2, batchCount2 = data.to_Vectors(trainSet2, userCount2, itemCount2)
CF3=Algo.CFGAN(trainSet2, userCount2, itemCount2, trainVector2, userMaskDict2, batchCount2, 10,0.5,0.7,0.03)
CF3.train()
app = Flask(__name__)
@app.route('/testGet', methods=['GET'])
def test():
    print("一次get")
    return "666"
@app.route('/recommend/movie',methods=['POST'])
def recommend_movie():
    try:
        user_id = int(request.form['user_id'])
    except Exception:
        return "无参数user_id"
    L=CF1.getMovieList(user_id,10)
    res={"error_code":"0","data":""}
    if(L==1):res["error_code"]=1
    elif(L==2):res["error_code"]=2
    else:res["data"]=L
    return str(res).replace("'",'"')
@app.route('/recommend/book',methods=['POST'])
def recommend_book():
    try:
        user_id = int(request.form['user_id'])
    except Exception:
        return "无参数user_id"
    L=CF2.getMovieList(user_id,10)
    res={"error_code":"0","data":""}
    if(L==1):res["error_code"]=1
    elif(L==2):res["error_code"]=2
    else:res["data"]=L
    return str(res).replace("'",'"')
@app.route('/recommend/music',methods=['POST'])
def recommend_music():
    try:
        user_id = int(request.form['user_id'])
    except Exception:
        return "无参数user_id"
    L=CF3.getMovieList(user_id,10)
    res={"error_code":"0","data":""}
    if(L==1):res["error_code"]=1
    elif(L==2):res["error_code"]=2
    else:res["data"]=L
    return str(res).replace("'",'"')
if __name__ == '__main__':

    app.run("0.0.0.0",debug=True,port=62001,use_reloader=False)