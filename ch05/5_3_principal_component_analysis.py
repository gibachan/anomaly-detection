# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# 観測データを取得
df = pd.read_csv("Cars93.csv")
mask = ["Min.Price", "Price", "Max.Price", "MPG.city", "MPG.highway", "EngineSize", 
    "Horsepower", "RPM", "Rev.per.mile", "Fuel.tank.capacity","Length", "Wheelbase",
    "Width", "Turn.circle", "Weight"]
X = df[mask]
X = (X - X.mean()) / X.std()    # 標準化
X = X.T                         # 転置
Xc = X.as_matrix()

# 散布行列の作成
S = Xc.dot(Xc.T)  

# 固有値・固有ベクトルを計算
evd, v = np.linalg.eig(S)

# 固有値をプロット -> エルボー則によりmを決定する
plt.plot(range(len(evd)), evd, color="red")
plt.title("Eigen values")
plt.xlabel("Eigen value number")
plt.ylabel("Eigen value")
plt.show()

# データクレンジングのため、訓練データに対して異常度を計算
m = 2                       # プロットの結果から、エルボー則により決定
x2 = v[:,0:m].T.dot(Xc)     # 正常部分空間内の成分を計算
a1 = (Xc * Xc).sum(axis=0) - (x2 * x2).sum(axis=0)  # 異常度を全訓練標本に対し計算

# 異常度上位6つを出力
result = pd.DataFrame({ "name": df["Make"], "a": a1 })
print(result.sort_index(by='a', ascending=False)[:6])               

# Car93データの正常部分空間における分布をプロット
plt.scatter(x2[0,:], x2[1,:], color="red")
for i in range(len(x2[0,:])):
    plt.text(x2[0,i], x2[1,i], i+1, ha="center", va="top", size=15)

plt.title("Cars93")
plt.xlabel("Second element")
plt.ylabel("First element")
plt.show()
