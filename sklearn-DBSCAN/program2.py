import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# 圆环分布点 + 0.05噪声
x1, y1 = datasets.make_circles(n_samples=2000, factor=0.5, noise=0.05)
# 圆形分布点 + 0.05噪声
x2, y2 = datasets.make_blobs(n_samples=1000, centers=[[1.2, 1.2]], cluster_std=[[.1]])

# 合并数据点
x = np.concatenate((x1, x2))
plt.scatter(x[:, 0], x[:, 1], marker='o')
plt.show()

# 使用K-MEANS进行聚合
y_pred = KMeans(n_clusters=3).fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.show()

# 使用DB-SCAN进行聚合
y_pred = DBSCAN(eps=0.2, min_samples=8).fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.show()
