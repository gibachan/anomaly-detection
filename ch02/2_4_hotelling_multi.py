# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import chi2

# 観測データを取得
df = pd.read_csv("Davis.csv")
weight = df["weight"]
height = df["height"]

# 体重-身長をプロット
plt.plot(weight, height, "bo")
plt.title("weight vs height")
plt.xlabel("weight")
plt.ylabel("height")
plt.show()

X = pd.concat([weight, height], axis=1) # 標本数 * 次元数(2)のデータ行列
X = X.as_matrix() # numpyのarray型へ変換
mx = X.mean(axis=0) # 標本平均
Xc = X - mx # 中心化したデータ行列
# Sx = (1.0 / len(X)) * Xc.T.dot(Xc) # 標本共分散行列
Sx = np.cov(X, rowvar=0, bias=1) # 標本共分散行列
a = (Xc.dot(np.linalg.pinv(Sx)) * Xc).sum(axis=1)

# 閾値を設定
th = chi2.isf(1 - 0.99, 2)  # 自由度2のカイ２乗分布

# 異常度をプロット
plt.plot(range(len(a)), a, "bo")
plt.axhline(y=th,color='red')
plt.show()


