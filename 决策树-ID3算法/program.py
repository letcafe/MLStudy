from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn import preprocessing
import csv

# 读入数据
Dtree = open(r'AllElectronics.csv')
reader = csv.reader(Dtree)

# 获取第一行数据(打印表头)
headers = reader.__next__()
print(headers)

# 定义两个列表
featureList = []
labelList = []

for row in reader:
    # 把label存入list
    labelList.append(row[-1])
    rowDict = {}
    for i in range(1, len(row) - 1):
        # 建立一个数据字典
        rowDict[headers[i]] = row[i]
    # 把数据字典存入list
    featureList.append(rowDict)

print("featureList: " + str(featureList))

# 把数据转换成01表示
vec = DictVectorizer()
x_data = vec.fit_transform(featureList).toarray()
print("x_data: \n" + str(x_data))

# 打印属性名称
print("vec.get_feature_names(): " + str(vec.get_feature_names()))

# 打印标签
print("labelList: " + str(labelList))

# 把标签转换成01表示
lb = preprocessing.LabelBinarizer()
y_data = lb.fit_transform(labelList)
print("y_data: " + str(y_data))

# 创建决策树模型，使用的策略为entropy(信息熵)，即ID3或C4.5的方式
model = tree.DecisionTreeClassifier(criterion='entropy')
# 输入数据建立模型
model.fit(x_data, y_data)

# 测试
x_test = x_data[0]
print("x_test: " + str(x_test))

predict = model.predict(x_test.reshape(1, -1))
print("predict: " + str(predict))

# 开始绘制图线
# 导出决策树
# pip install graphviz
# http://www.graphviz.org/
import graphviz

dot_data = tree.export_graphviz(model,
                                out_file=None,
                                feature_names=vec.get_feature_names(),
                                class_names=lb.classes_,
                                filled=True,
                                rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('computer')
