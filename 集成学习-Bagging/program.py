from sklearn import neighbors
from sklearn import datasets
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x_data = iris.data[:, :2]
y_data = iris.target

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

knn = neighbors.KNeighborsClassifier()
knn.fit(x_train, y_train)


# 将绘图函数封装，传入训练的模型即可
def plot(model):
    # 获取数据值所在的范围
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1

    # 生成网格矩阵
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    z = model.predict(np.c_[xx.ravel(), yy.ravel()])  # ravel与flatten类似，多维数据转一维。flatten不会改变原始数据，ravel会改变原始数据
    z = z.reshape(xx.shape)
    # 等高线图
    cs = plt.contourf(xx, yy, z)


# 画图
plot(knn)
# 样本散点图
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()
# 准确率
print("KNN only score = " + str(knn.score(x_test, y_test)))

dtree = tree.DecisionTreeClassifier()
dtree.fit(x_train, y_train)

# 画图
plot(dtree)
# 样本散点图
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()
# 准确率
print("DTree only score = " + str(dtree.score(x_test, y_test)))

# 随机抽样到100个bag，用100个bag进行投票（KNN）
bagging_knn = BaggingClassifier(knn, n_estimators=100)
# 输入数据建立模型
bagging_knn.fit(x_train, y_train)
plot(bagging_knn)
# 样本散点图
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()
print("Bagging KNN score = " + str(bagging_knn.score(x_test, y_test)))

# 随机抽样到100个bag，用100个bag进行投票（决策树）
bagging_tree = BaggingClassifier(dtree, n_estimators=100)
# 输入数据建立模型
bagging_tree.fit(x_train, y_train)
plot(bagging_tree)
# 样本散点图
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()
print("Bagging DTree only score = " + str(bagging_tree.score(x_test, y_test)))

print("program ended")
