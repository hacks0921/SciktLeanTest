import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import pydot


# 1)데이터 불러오기
df = pd.read_csv("data/diabetes2.csv", encoding='CP949')
print(df)


# 2)X 컬럼 선정하기
print(df.columns)
X = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI','DiabetesPedigreeFunction', 'Age']]
print(X.shape)


## Min MAX Scalling
from sklearn.preprocessing import MinMaxScaler              # MAX - Min
min_max_scaler = MinMaxScaler()
X_MinMax_train = min_max_scaler.fit_transform(X)
print(X_MinMax_train)
X_MinMax_train  = pd.DataFrame(X_MinMax_train, columns=X.columns)
print(X_MinMax_train)

## Standard  Scalling
from sklearn.preprocessing import StandardScaler            # 평균 0 , 표준편차 1
sdscaler = StandardScaler()
X_sdscaler_train = sdscaler.fit_transform(X)
X_sdscaler_train  = pd.DataFrame(X_sdscaler_train, columns=X.columns)
print(X_sdscaler_train)



# # 2)Y 컬럼 선정하기
# y = df['Outcome']
# print(df.mean)
# answer = df['Outcome'].value_counts()
#
# answer_values = answer.values
# answer_keys = answer.keys
# print("*"*100)
# print(answer_values)
# print(answer_keys)
# answer_keys = answer.keys()
# answer_keys = answer.keys()
#
# print("*"*100)
# print(answer_values)
# print(answer_keys)
# answer_keys = answer.keys().tolist()
# answer_keys = answer.keys().tolist()
#
# print("*"*100)
# print(answer_values)
# print(answer_keys)
