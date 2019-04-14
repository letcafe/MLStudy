import numpy as np
from numpy import genfromtxt
from sklearn import linear_model

# 读入数据
data = genfromtxt(r"longley.csv", delimiter=',')
print(data)

# 切分数据
x_data = data[1:, 2:]
y_data = data[1:, 1]
print(x_data)
print(y_data)

# 创建模型
model = linear_model.ElasticNetCV(cv=4)
model.fit(x_data, y_data)

# 弹性网系数
print(model.alpha_)
# 相关系数
print(model.coef_)

val = model.predict(x_data[2, np.newaxis])
print(val)

# predict = 88.33171474, raw = 88.2