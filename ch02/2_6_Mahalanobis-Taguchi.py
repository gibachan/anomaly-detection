# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import chi2

# 観測データを取得
df = pd.read_csv("road.csv", index_col=0)   # インデックス列は除いて読み込む

# Numpyのarray型へ変換
X = df.drop("drivers", axis=1).as_matrix()  # drivers列を除く
drivers = df.as_matrix(columns=["drivers"])

# マハラノビス=タグチ法
X = X / drivers     # 1ドライバー当たりの数値に変換
X = np.log(X + 1)   # 対数変換（ボックス=コックス変換）
mx = X.mean(axis=0) # 標本平均
Xc = X - mx         # 中心化したデータ行列
# Sx = (1.0 / len(X)) * Xc.T.dot(Xc) # 標本共分散行列
Sx = np.cov(X, rowvar=0, bias=1) # 標本共分散行列
a = (Xc.dot(np.linalg.pinv(Sx)) * Xc).sum(axis=1) / X.shape[1]  # 1変数当たりの異常度

# 閾値を決定
# （標本が正常範囲に入るように1変数当たりのマハラノビス距離の閾値を決める）
th = 1.0

state_label = df.index[a>th]    # 閾値を超えた州の名前リスト
state_a = a[a>th]               # 閾値を超えた州の異常度リスト
print(state_label)
print(state_a)

# プロット
plt.plot(range(len(a)), a, "bo")
plt.axhline(y=th,color='red')
plt.title("Anomaly score")
plt.xlabel("Sample number")
plt.ylabel("Anomaly score")
plt.show()

# SN比解析
xc_prime = Xc[4,:]  # 中心化行列からCalifのデータ行を取得
SN1 = 10 * np.log10(xc_prime**2 / np.diag(Sx))

plt.bar(range(len(SN1)), SN1, tick_label=["deaths","popden","rural","temp","fuel"], align="center")
plt.title("SN ratio")
plt.show()
