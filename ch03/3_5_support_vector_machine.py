# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn import svm
from scipy import stats

# サンプルデータを作成
X1 = np.random.normal(0.0, 1.0, (60,2)) # 平均0.0、分散1.0の正規分布に乱数を60*2行列で作成
X2 = np.random.normal(3.0, 1.0, (60,2)) # 平均3.0、分散1.0の正規分布に乱数を60*2行列で作成
X = np.vstack([X1, X2]) # X1とX2を連結

# 標準化（平均0、分散1となるように変換）
X_train = preprocessing.scale(X)

# SVM
clf = svm.OneClassSVM(nu=0.5, kernel="rbf", gamma=0.5)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)

# 閾値を設定
th = stats.scoreatpercentile(y_pred_train, 100 * 0.05) # パーセンタイルで異常判定の閾値設定
        
# ２次元作図用格子状データの生成
xx, yy = np.meshgrid(np.linspace(-4, 4, 500), np.linspace(-4, 4, 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) # 格子状に超平面との距離を出力
Z = Z.reshape(xx.shape)

# プロット
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), th, 7), cmap=plt.cm.Blues_r)
plt.contour(xx, yy, Z, levels=[th], linewidths=2, colors='green')
plt.scatter(X_train[y_pred_train==1][:,0], X_train[y_pred_train==1][:,1], color="red")
plt.scatter(X_train[y_pred_train==-1][:,0], X_train[y_pred_train==-1][:,1], color="blue")
plt.show()
