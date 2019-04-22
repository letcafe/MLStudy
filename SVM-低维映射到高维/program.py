import matplotlib.pyplot as plt
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D

# 生成圆环数据集
x_data, y_data = datasets.make_circles(n_samples=500, factor=.3, noise=.10)
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
# 画出二维数据集
plt.show()

# 升维操作：新的维度变量=a**2+b**2
z_data = x_data[:, 0] ** 2 + x_data[:, 1] ** 2

# 绘制升维之后的空间图
ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(x_data[:, 0], x_data[:, 1], z_data, c=y_data, s=10)  # 点为红色三角形

# 显示图像
plt.show()
