# anomaly-detection

[入門 機械学習による異常検知―Rによる実践ガイド ](http://www.coronasha.co.jp/np/isbn/9784339024913/)をPythonで実装

## 2 正規分布に従うデータからの異常検知

ホテリング理論に基づく外れ値検知。

### 2.2 1変数正規分布のに基づく異常検知

観測データが単一の正規分布に従うと仮定した場合の、確率分布のパラメータを最尤推定法で推定する。

### 2.4 多変量正規分布に基づく異常検知

複数の変数が独立に正規分布に従うと仮定した場合。

### 2.6 マハラノビス=タグチ法

変数毎の異常度への影響度をSN値で表す。

## 3.非正規データからの異常検知

正規分布以外の分布に従うデータに対する外れ値検知。

### 3.1 分布が左右対称でない場合

データをガンマ分布に当てはめて考える。ガンマ分布の確率密度関数のパラメータ（k,s）をモーメント法で推定する。

### 3.2 訓練データに異常表本が混ざっている場合

正常標本と異常標本の混合正規分布から、期待値-最大化法（EM法）によりそれぞれの正規分布のパラメータを推測する。

