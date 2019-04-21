from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

x = [[3, 3], [4, 3], [1, 1]]
y = [1, 1, -1]

x_data = np.array(x)[:, 0]
y_data = np.array(x)[:, 1]
plt.scatter(x_data, y_data, c='r')
plt.show()

model = svm.SVC(kernel='linear')
model.fit(x, y)

# 打印支持向量
print("model.support_vectors_ = " + str(model.support_vectors_))
# 第2和第0个点是支持向量
print("model.support_ = " + str(model.support_))
# 有几个支持向量
print("model.n_support_ = " + str(model.n_support_))
print("model.predict([[4, 3]]) = " + str(model.predict([[4, 3]])))

# 构建出来的超平面的属性，本例中形如：ax1+bx2+d=0
# 参数斜率,a=0.5,b=0.5
print("model.coef_ = " + str(model.coef_))
# 参数偏置（截距）,d=4
print("model.intercept_ = " + str(model.intercept_))
# 可见分类直线: 0.5x+0.5y-2=0
