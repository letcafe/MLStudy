import numpy as np
import matplotlib.pyplot as plt

# 输入数据：异或计算
# 0列为w0对应的1
# 1列为x对应的值
# 2列为y对应的值
# 3列为x^2对应的值
# 4列为xy对应的值
# 5列为y^2对应的值
# 公式：X.dot(W.T) = Y
# 即：w0 + w1*x + w2*y + w3*x1^2 + w4*x*y + w5*y^2
X = np.array([
    [1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 1],
    [1, 1, 0, 1, 0, 0],
    [1, 1, 1, 1, 1, 1]
])

# 标签
Y = np.array([-1, 1, 1, -1])
# 权值初始化随机数，1行6列，取值范围为-1到1
W = (np.random.random(6) - 0.5) * 2
print(W)
# 学习率设置
lr = 0.11
# 计算迭代次数
n = 0
# 神经网络输出
O = 0


def update():
    global X, Y, W, lr, n
    n += 1
    O = np.dot(X, W.T)
    W_C = lr * ((Y - O.T).dot(X)) / int(X.shape[0])
    W = W + W_C


for _ in range(10000):
    # 更新权值
    update()

# 正样本
x1 = [0, 1]
y1 = [1, 0]
# 负样本
x2 = [0, 1]
y2 = [0, 1]


def calculate(x, root):
    a = W[5]
    b = W[2] + x * W[4]
    c = W[0] + x * W[1] + x * x * W[3]
    if root == 1:
        return (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    if root == 2:
        return (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)


xdata = np.linspace(-1, 2)
print(xdata)

plt.figure()

# 分别绘制两个二元方程结果的曲线
plt.plot(xdata, calculate(xdata, 1), c='r')
plt.plot(xdata, calculate(xdata, 2), c='r')

# 画出样本点
plt.plot(x1, y1, 'bo')
plt.plot(x2, y2, 'yo')
plt.show()

print(W)

O = np.dot(X, W.T)
print(O)
