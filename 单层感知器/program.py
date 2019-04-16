import numpy as np
import matplotlib.pyplot as plt

# 输入数据
X = np.array([
    [1, 3, 3],
    [1, 4, 3],
    [1, 1, 1],
    [1, 0, 2]
])

# 标签
Y = np.array([
    [1],
    [1],
    [-1],
    [-1]
])

# 权值初始化，3行1列，取值范围为-1到1
W = (np.random.random([3, 1]) - 0.5) * 2

# W = [[ 0.54750753]
#  [ 0.64967354]
#  [-0.3031968 ]]


print(W)

# 学习率设置
lr = 0.11
# 神经网络输出
O = 0


def update():
    global X, Y, W, lr
    O = np.sign(np.dot(X, W))
    # X.T为转置矩阵，shape[0]:(3,1)
    W_C = lr * (X.T.dot(Y - O)) / int(X.shape[0])
    W = W + W_C


for i in range(100):
    update()
    print(W)
    print(i)
    # 计算当前输出
    O = np.sign(np.dot(X, W))
    # 如果实际输出等于期望输出，模型收敛，循环结束
    if (O == Y).all():
        print('Finished')
        print('epoch:', i)
        break

# 正样本
x1 = [3, 4]
y1 = [3, 3]

# 负样本
x2 = [1, 0]
y2 = [1, 2]

# 计算分界线的斜率以及截距
k = -W[1] / W[2]
d = -W[0] / W[2]
print('k=', k)
print('d=', d)

# 定义两个点，用于画图
xdata = (0, 5)

# 画预测线
plt.figure()
plt.plot(xdata, xdata * k + d, 'r')

# 画出点
plt.scatter(x1, y1, c='b')
plt.scatter(x2, y2, c='y')
plt.show()
