import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

#データをロードし、labelとfeaturesそれぞれを取得
df = pd.read_csv('winequality-red.csv', delimiter=";")
X_train, X_test, y_train, y_test = train_test_split(df.drop('quality', axis=1), df['quality'], test_size=0.25, random_state=1)

#線形回帰モデルを作成し、ワインデータを使って学習をする
regr = linear_model.RidgeCV(alphas= np.arange(0.1,10.0,.5))
regr.fit(X_train, y_train)

#モデルをシリアライズしファイル名「winemodel.pkl」で保存
pickle.dump(regr,open("winemodel.pkl","wb"))

#エラーチェック
ans = regr.predict(X_test)
print mean_squared_error(y_test, ans)
