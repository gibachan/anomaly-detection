# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# 観測データを取得
df = pd.read_csv("qtdbse1102.txt", delimiter="\t")
X = df.ix[:,1]  # 一列だけを取り出す
X = X.as_matrix()

# 変数を定義
w = 100 # 窓幅
nk = 1  # 近傍数
T = 3000 # 観測値の長さ

# 前半を訓練データ、残りを検証データとする
Xtr = X[:T]
X = X[T+1:2*T+1]

# スライド窓で分割した時系列データを作成
def create_slide_data(data, w):
    D = []
    T = len(data)
    N = T - w + 1
    for i in range(N):
        D.append(data[i:i+w])
    return D

# 2点間の距離（ユークリッド距離）　
def dist(pt1, pt2):
    return np.sqrt(((pt2 - pt1) ** 2).sum())

# 異常度を計算
# 距離のうち最小のものをk個選び、平均を計算する
def score(dist_list, k):
    dist_list.sort()
    return sum(dist_list[:k]) / k

# 時系列データを作成
Dtr = create_slide_data(Xtr, w) # 訓練データ
D = create_slide_data(X, w)     # 検証データ

# 異常度を計算
# この実装では計算時間が多くかかってしまう
a_list = []
for x in D:
    dist_list = []
    for xtr in Dtr:
        dist_list.append(dist(x, xtr))
    a = score(dist_list, nk)
    a_list.append(a)
    

# プロット
plt.plot(range(len(a_list)), a_list, linestyle="solid", color="red")
plt.title("Anomaly score")
plt.xlabel("Index")
plt.ylabel("Anomaly score")
plt.show()

