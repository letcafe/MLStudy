import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 创建40个点
# 生成随机20个点，每个点减去(2,2)，在随机生成20个点，每个点加上(2,2)，共40个点，分成了两片
x_data = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
# y总共40个标签值，前20个是0，后20个是1
y_data = [0] * 20 + [1] * 20
# 画出点图
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()

# 选择线性SVM拟合
model = svm.SVC(kernel='linear')
model.fit(x_data, y_data)
# 即方程为：model.coef_[0][0] * x + model.coef_[0][1] * y + model.intercept_ = 0
print("系数项 = " + str(model.coef_))
print("偏置项/截距 = " + str(model.intercept_))

# 根据分离平面加上点画出分离平面（本例为直线）
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
x_test = np.array([[-5], [5]])
d = -model.intercept_ / model.coef_[0][1]
k = -model.coef_[0][0] / model.coef_[0][1]
y_test = d + k * x_test
plt.plot(x_test, y_test, 'k')
plt.show()

# 画出通过支持向量的分界线（图中的虚线），注意细节，第0索引和第-1索引是不一样的两个支持向量，蓝线和红线
b1 = model.support_vectors_[0]
y_down = k * x_test + (b1[1] - k * b1[0])
b2 = model.support_vectors_[-1]
y_up = k * x_test + (b2[1] - k * b2[0])

plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
x_test = np.array([[-5], [5]])
d = -model.intercept_ / model.coef_[0][1]
k = -model.coef_[0][0] / model.coef_[0][1]
y_test = d + k * x_test
# 画出分离平面以及两个边界平面
plt.plot(x_test, y_test, 'k')
plt.plot(x_test, y_down, 'r--')
plt.plot(x_test, y_up, 'b--')
plt.show()
