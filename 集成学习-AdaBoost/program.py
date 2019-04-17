import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.metrics import classification_report

# 该数据集由sklearn随机生成
# 生成2维正态分布，生成的数据按分位数分为两类，500个样本,2个样本特征,样本特征均值为0,0
x1, y1 = make_gaussian_quantiles(mean=(0, 0), n_samples=500, n_features=2, n_classes=2)
# 生成2维正态分布，生成的数据按分位数分为两类，400个样本,2个样本特征，样本特征均值为3,3
x2, y2 = make_gaussian_quantiles(mean=(3, 3), n_samples=500, n_features=2, n_classes=2)
# 将两组数据合成一组数据
x_data = np.concatenate((x1, x2))
# - y2 + 1,由于y是0,1的特征矩阵，该方式可以将0,1进行互换
y_data = np.concatenate((y1, 1 - y2))

# 画图测试，可以发现生成的点原来的图是内黑色外黄色
plt.scatter(x1[:, 0], x1[:, 1], c=y1)
plt.show()

# 经过转换，得到了两团正态分布点，一团内黑外黄(0,0)，一团内黄外黑(3,3)
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()

# 算法将使用决策树与Adaboost进行对比
# 决策树模型
model = tree.DecisionTreeClassifier(max_depth=3)

# 输入数据建立模型
model.fit(x_data, y_data)

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
# 样本散点图
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()

# 模型准确率
print("DTree model score = " + str(model.score(x_data, y_data)))

# AdaBoost模型(弱学习器使用决策树模型（决策树的深度 = 3），迭代次数为10)
model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=10)
# 训练模型
model.fit(x_data, y_data)

# 获取数据值所在的范围
x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1

# 生成网格矩阵
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 获取预测值
z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
# 等高线图
cs = plt.contourf(xx, yy, z)
# 样本散点图
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()

# 模型准确率
model.score(x_data, y_data)

print("AdaBoost model score = " + str(model.score(x_data, y_data)))
