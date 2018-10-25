import pickle
import flask
from flask import request

app = flask.Flask(__name__)

#学習済みモデルをロード
model = pickle.load(open("winemodel.pkl","r"))

#defining a route for only post requests
@app.route('/predict', methods=['POST'])
def index():
    #リクエストに含まれている、ワインの特徴情報(features)を取得
    feature_array = request.get_json()['feature_array']

    #モデルにワインの特徴情報を渡し予測
    #レスポンス情報を準備
    response = {}
    response['predictions'] = model.predict([feature_array]).tolist()

    #レスポンスを返す（JSON）
    return flask.jsonify(response)
