# -*- coding: utf-8 -*-

# 標本の分布をガンマ分布に当てはめて異常を計算する 

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# データ読み込み
df = pd.read_csv("Davis.csv")
weight = df["weight"].as_matrix()

# ガンマ分布における異常度を計算
N = len(weight)     # 標本数
mu = weight.mean()  # 標本平均
si = np.std(weight) # 標準偏差　
kmo = (mu / si) ** 2    # モーメント法によるkの推定値
smo = (si ** 2) / mu    # モーメント法によるsの推定値

a = weight / smo - (kmo - 1) * np.log(weight / smo)    # 異常度
sorted_a = np.sort(a)[::-1] # 降順にソート　
th = sorted_a[(int(N * 0.01 - 1))]  # 異常度の上位1%分位点に置ける異常度

# プロット
plt.plot(range(len(a)), a, "bo")
plt.axhline(y=th,color='red')
plt.title("Anomaly score")
plt.xlabel("Sample number")
plt.ylabel("Anomaly score")
plt.show()