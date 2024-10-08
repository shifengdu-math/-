import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV  #训练集测试集划分，交叉验证
from sklearn.feature_extraction import DictVectorizer  #特征选择
from sklearn.ensemble import RandomForestClassifier    #模型

titanic = pd.read_csv(fr'C:\Users\shifeng.du\Desktop\github\sklearn-model\data\titanic.csv')

x = titanic[['Pclass', 'Age', 'Sex']]
y = titanic['Survived']

x["Age"].fillna(x["Age"].mean(), inplace=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)

transfer = DictVectorizer(sparse=False) 
x_train = transfer.fit_transform(x_train.to_dict(orient="records"))
x_test = transfer.fit_transform(x_test.to_dict(orient="records"))

rfc = RandomForestClassifier()
param = {"n_estimators": [120,200,300,500,800,1200], "max_depth": [5, 8, 15, 25, 30]}
gc = GridSearchCV(rfc, param_grid=param, cv=2) 
gc.fit(x_train, y_train) 
print("随机森林预测的准确率为：", gc.score(x_test, y_test))
