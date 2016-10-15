# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.mixture import GMM

# 観測データを取得
df = pd.read_csv("Davis.csv")
weight = df["weight"]
height = df["height"]

X = pd.concat([weight, height], axis=1) # 標本数 * 次元数(2)のデータ行列
X_train = X.drop(11)  # 11番目のデータを除去
X = X.as_matrix()
X_train = X_train.as_matrix()

# 混合正規分布(ガウス混合分布)モデルによるクラスタリング
clf = GMM(n_components=2)
clf.fit(X_train)
X_pred = clf.predict(X_train)

# クラスタリングの結果をプロット
cluster0 = X_train[X_pred == 0]
cluster1 = X_train[X_pred == 1]
plt.scatter(cluster0[:,0], cluster0[:,1], color="red")
plt.scatter(cluster1[:,0], cluster1[:,1], color="blue")
plt.show()

# 混合比を取り出す
pi = clf.weights_

# 異常度を計算
anomaly_scores = -clf.score_samples(X)[0]

# プロット
plt.title("Anomaly score")
plt.xlabel("Sample number")
plt.ylabel("Anomaly score")
plt.plot(range(len(anomaly_scores)), anomaly_scores, "bo")
plt.show()
