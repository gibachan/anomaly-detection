# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

# 観測データを取得
df = pd.read_csv("Davis.csv")
weight = df["weight"]
height = df["height"]

# カーネル密度推定
values = np.vstack([weight, height])
kernel = gaussian_kde(values)

# カーネル密度推定結果をグラフ化
weight_max = weight.max()
weight_min = weight.min()
height_max = height.max()
height_min = height.min()
X, Y = np.mgrid[weight_min:weight_max:100j, height_min:height_max:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(kernel(positions).T, X.shape)

fig, ax = plt.subplots()
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[weight_min, weight_max, height_min, height_max])
ax.plot(weight, height, 'k.', markersize=2)
ax.set_xlim([weight_min, weight_max])
ax.set_ylim([height_min, height_max])
plt.show()

# 異常度を計算
a = -kernel.logpdf(values)
plt.plot(range(len(a)), a, "bo")
plt.show()

# 異常度が高いサンプルを抽出
print(values[:,a>10].T)


