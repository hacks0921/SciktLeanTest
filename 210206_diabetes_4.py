import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/diabetes_feature.csv")
print(df.shape)

print(df.columns)
# X = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Pregnancies_high', 'Age_low', 'Age_middle', 'Insulin_nan', 'low_glu_insulin']]
X = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI','DiabetesPedigreeFunction', 'Age','Pregnancies_high']]
print(X.shape)
y = df['Outcome']
print(y.shape)


from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# X : Featuer, y: lable 값

# 테스트셋을 test_size 옵션을 통해 20%로 지정합니다.
# 매번 샘플링할 때마다 데이터가 달라질 수 있으므로 random_state=42로 같은 것을 사용합니다.

from sklearn.tree import DecisionTreeClassifier
#
# model = DecisionTreeClassifier(max_depth=11, random_state=42)
# model.fit(X_train, y_train)
# y_predict = model.predict(X_test)
# print(y_predict)
# print(abs(y_predict - y_test).sum())

from sklearn.metrics import accuracy_score

# accuracy_score = accuracy_score(y_test, y_predict) * 100
# print(accuracy_score)

# for max_depth in range(3, 12):
#     model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
#     y_predict = model.fit(X_train, y_train).predict(X_test)
#     score = accuracy_score(y_test, y_predict) * 100
#     print(max_depth, score)


from sklearn.model_selection import GridSearchCV

model = DecisionTreeClassifier(random_state=42)
param_grid = {"max_depth": range(3, 12),
"max_features": [0.3, 0.5, 0.7, 0.9, 1]}
clf = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, cv=5, verbose=2)
clf.fit(X_train, y_train)