from sklearn import neighbors
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 载入数据
iris = datasets.load_iris()
print(iris)
# 打乱数据切分数据集
# 分割数据0.2为测试数据，0.8为训练数据
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# 构建模型(n_neighbors即为k)
model = neighbors.KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
prediction = model.predict(x_test)

print(classification_report(y_test, prediction))
