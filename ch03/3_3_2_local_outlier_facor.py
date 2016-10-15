# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math

# 観測データを取得
df = pd.read_csv("Davis.csv")
weight = df["weight"]
height = df["height"]

X = pd.concat([weight, height], axis=1) # 標本数 * 次元数(2)のデータ行列
X = X.as_matrix()

# 2点間の距離を計算
def calc_distance(x1, x2):
    return math.sqrt((x2[0] - x1[0]) ** 2 + (x2[1] - x1[1]) ** 2)

# targetからの距離によってdataをソートする
def sort_by_distance(data, target):
    tmp_list = list(data)
    tmp_list.sort(key=lambda x:calc_distance(target, x))
    return np.array(tmp_list)

# k近傍データと最小半径Ekを計算
# data: targetを含まないデータ
def get_k_nearest_neighbers_and_radius(k, data, target):
    sorted_data = sort_by_distance(data, target)
    return sorted_data[:k], calc_distance(target, sorted_data[:k][-1])

# 近傍有効距離を計算
# data: u, udashを含まないデータ
def calc_reachability_distance(k, data, u, udash):
    nearest_neighbers_u, Ek_u = get_k_nearest_neighbers_and_radius(k, data, u)
    nearest_neighbers_udash, Ek_udash = get_k_nearest_neighbers_and_radius(k, data, udash)
    dist = calc_distance(u, udash)
    lk_u_udash = 0.0
    if dist <= Ek_u and dist <= Ek_udash:
        lk_u_udash = Ek_udash
    else:
        lk_u_udash = dist
    return lk_u_udash
    
# targetの周りのk近傍にわたる近傍有効距離の平均値dk(target)を計算
# data: targetを含まないデータ
def calc_k_nearest_mean_distance(k, data, target):
    # targetのk近傍データを取得
    nearest_neighbers_target, Ek_target = get_k_nearest_neighbers_and_radius(k, data, target)
    total_lk = 0.0  # 近傍有効距離の合計値
    # 近傍点を走査
    for i, neighber in enumerate(nearest_neighbers_target):
        # データ全体から近傍点neighberを除く
        data_without_neighber = sort_by_distance(data, neighber)[1:]
        # 近傍有効距離を計算
        lk_target_neighber = calc_reachability_distance(k, data_without_neighber, target, neighber)
        # 合計値に加算
        total_lk += lk_target_neighber
    return total_lk / k # 平均値を計算 


# 局所外れ値度に基づくk近傍法による異常検知
k = 5
aLOF = []   # サンプル毎の異常値リスト

# サンプルの個々のデータを走査
for i, p in enumerate(X):
    # データ全体からpを除く
    data_without_p = np.delete(X, i, axis=0)    

    # dk(p)を計算
    dk_p = calc_k_nearest_mean_distance(k, data_without_p, p)

    # pのk近傍データを取得
    neighbers_p, Ek_p = get_k_nearest_neighbers_and_radius(k, data_without_p, p)

    # 異常度aLOF(p)を計算
    aLOF_p = 0.0
    for j, q in enumerate(neighbers_p):
        # データ全体からp, qを除く
        data_without_pq = sort_by_distance(data_without_p, q)[1:]
        
        # dk(q)を計算
        dk_q = calc_k_nearest_mean_distance(k, data_without_pq, q)
        if dk_q != 0.0:
            aLOF_p += (dk_p / dk_q)

    aLOF_p = aLOF_p / k
    aLOF.append(aLOF_p)

# 0.01パーセンタイルで閾値を設定
sorted_a = np.sort(aLOF)[::-1]         # 降順にソート　
th = sorted_a[(int(len(aLOF) * 0.01 - 1))]  # 異常度の上位1%分位点に置ける異常度

# プロット
plt.title("Anomaly score")
plt.xlabel("Sample number")
plt.ylabel("Anomaly score")
plt.plot(range(len(aLOF)), aLOF, "bo")
plt.axhline(y=th,color='red')
plt.show()


    



