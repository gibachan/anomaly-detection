# -*- coding: utf-8 -*-

# 正常標本と異常標本の混合正規分布から、期待値-最大化法（EM法）によりそれぞれの正規分布のパラメータを推測する。

import numpy as np
from scipy.stats import norm
import math
from matplotlib import pyplot as plt

N = 1000

mu0 = 3.0
mu1 = 0.0
sig0 = 0.5
sig1 = 3.0
pi0 = 0.6
pi1 = 0.4

n0 = norm.rvs(mu0, sig0, size=N*pi0)
n1 = norm.rvs(mu1, sig1, size=N*pi1)
n = np.concatenate([n0, n1])
np.random.shuffle(n)            # シャッフル

# 期待値-最大化法
ite = range(20)
pi0 = 0.5
pi1 = 0.5
mu0 = 5.0
mu1 = -5.0
sig0 = 1.0
sig1 = 5.0

pi0_list = []
pi1_list = []
mu0_list = []
mu1_list = []
sig0_list = []
sig1_list = []

for i in ite:
    piN0 = norm.pdf(x=n, loc=mu0, scale=sig0)
    piN1 = norm.pdf(x=n, loc=mu1, scale=sig1)
    qn0 = piN0 / (piN0 + piN1)
    qn1 = piN1 / (piN0 + piN1)

    pi0 = qn0.sum() / N
    pi1 = qn1.sum() / N
    mu0 = (qn0 * n).sum() / (N * pi0)
    mu1 = (qn1 * n).sum() / (N * pi1)
    sig0 = math.sqrt((qn0 * (n - mu0) * (n - mu0)).sum() / (N * pi0))
    sig1 = math.sqrt((qn1 * (n - mu1) * (n - mu1)).sum() / (N * pi1))

    pi0_list.append(pi0)
    pi1_list.append(pi1)
    mu0_list.append(mu0)
    mu1_list.append(mu1)
    sig0_list.append(sig0)
    sig1_list.append(sig1)


print("正常モデル pi={0}, mu={1}, sig={2}".format(pi0, mu0, sig0))
print("雑音モデル pi={0}, mu={1}, sig={2}".format(pi1, mu1, sig1))

# プロット
plt.subplot(1, 6, 1)
plt.title("pi0")
plt.xlabel("iteration")
plt.ylabel("pi0")
plt.plot(ite, pi0_list)

plt.subplot(1, 6, 2)
plt.title("pi1")
plt.xlabel("iteration")
plt.ylabel("pi1")
plt.plot(ite, pi1_list)

plt.subplot(1, 6, 3)
plt.title("mu0")
plt.xlabel("iteration")
plt.ylabel("mu0")
plt.plot(ite, mu0_list)

plt.subplot(1, 6, 4)
plt.title("mu1")
plt.xlabel("iteration")
plt.ylabel("mu1")
plt.plot(ite, mu1_list)

plt.subplot(1, 6, 5)
plt.title("sig0")
plt.xlabel("iteration")
plt.ylabel("sig0")
plt.plot(ite, sig0_list)

plt.subplot(1, 6, 6)
plt.title("sig1")
plt.xlabel("iteration")
plt.ylabel("sig1")
plt.plot(ite, sig1_list)

plt.show()


