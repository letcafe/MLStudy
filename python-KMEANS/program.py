import numpy as np
import matplotlib.pyplot as plt

# 载入数据
data = np.genfromtxt("kmeans.txt", delimiter=" ")

plt.scatter(data[:, 0], data[:, 1])
plt.show()

print("data.shape = " + str(data.shape))


# 计算距离
def cal_distance(vector1, vector2):
    return np.sqrt(sum((vector2 - vector1) ** 2))


# 初始化质心
def init_central_points(data, k):
    num_samples, dim = data.shape
    # k个质心，列数跟样本的列数一样(k行2列)
    centroids = np.zeros((k, dim))
    # 随机选出k个质心
    for i in range(k):
        # 随机选取一个样本的索引
        index = int(np.random.uniform(0, num_samples))
        # 作为初始化的质心（将对应的坐标赋值给k个中心点）
        centroids[i, :] = data[index, :]
    return centroids


def kmeans(data, k):
    # 计算样本个数
    num_samples = data.shape[0]
    # 样本的属性，第一列保存该样本属于哪个簇，第二列保存该样本跟它所属簇的误差
    cluster_data = np.array(np.zeros((num_samples, 2)))
    cluster_changed = True

    # 初始化质心
    centroids = init_central_points(data, k)

    while cluster_changed:
        cluster_changed = False
        # 循环每一个样本
        for i in range(num_samples):
            # 最小距离
            min_dist = 1000000.0
            # 定义样本所属的簇
            min_index = 0
            # 循环计算每一个质心与该样本的距离
            for j in range(k):
                # 循环每一个质心和样本，计算距离
                distance = cal_distance(centroids[j, :], data[i, :])
                # 如果计算的距离小于最小距离，则更新最小距离
                if distance < min_dist:
                    # 更新最小距离
                    min_dist = distance
                    cluster_data[i, 1] = min_dist
                    # 更新样本所属的簇
                    min_index = j
            # 如果样本的所属的簇发生了变化
            if cluster_data[i, 0] != min_index:
                # 质心要重新计算
                cluster_changed = True
                # 更新样本的簇
                cluster_data[i, 0] = min_index

        # 更新质心
        for j in range(k):
            # 获取第j个簇所有样本所在的索引
            cluster_index = np.nonzero(cluster_data[:, 0] == j)
            # 第j个簇所有的样本点
            points_in_cluster = data[cluster_index]
            # 计算质心
            centroids[j, :] = np.mean(points_in_cluster, axis=0)
    return centroids, cluster_data


# 显示结果
def show_cluster(data, k, centroids, cluster_data):
    num_samples, dim = data.shape
    if dim != 2:
        print("dimension of your data is not 2!")
        return 1
    # 用不同颜色形状来表示各个类别
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("Your k is too large!")
        return 1
        # 画样本点
    for i in range(num_samples):
        mark_index = int(cluster_data[i, 0])
        plt.plot(data[i, 0], data[i, 1], mark[mark_index])
    # 用不同颜色形状来表示各个类别
    mark = ['*r', '*b', '*g', '*k', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 画质心点
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=20)
    plt.show()


# 设置k值
k = 4
# centroids 簇的中心点
# cluster Data样本的属性，第一列保存该样本属于哪个簇，第二列保存该样本跟它所属簇的误差
centroids, clusterData = kmeans(data, k)
if np.isnan(centroids).any():
    print('Error')
else:
    print('cluster complete!')
    # 显示结果
show_cluster(data, k, centroids, clusterData)

# # 做预测
# 做预测
x_test = [0, 1]
# 将x_test复制4次
np.tile(x_test, (k, 1))
# 误差
np.tile(x_test, (k, 1)) - centroids
# 误差平方
(np.tile(x_test, (k, 1)) - centroids) ** 2
print("predict_index = " + str((np.tile(x_test, (k, 1)) - centroids) ** 2))

# 误差平方和
((np.tile(x_test, (k, 1)) - centroids) ** 2).sum(axis=1)
print("predict_index = " + str(((np.tile(x_test, (k, 1)) - centroids) ** 2).sum(axis=1)))

# 最小值所在的索引号
predict_index = np.argmin(((np.tile(x_test, (k, 1)) - centroids) ** 2).sum(axis=1))
print("predict_index = " + str(predict_index))


def predict(datas):
    return np.array([np.argmin(((np.tile(data, (k, 1)) - centroids) ** 2).sum(axis=1)) for data in datas])


# # 画出簇的作用区域
# 获取数据值所在的范围
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1

# 生成网格矩阵
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

z = predict(np.c_[xx.ravel(), yy.ravel()])
# ravel与flatten类似，多维数据转一维。flatten不会改变原始数据，ravel会改变原始数据
z = z.reshape(xx.shape)
# 等高线图
cs = plt.contourf(xx, yy, z)
# 显示结果
show_cluster(data, k, centroids, clusterData)
