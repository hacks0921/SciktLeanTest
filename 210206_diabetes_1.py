import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import sys


# 1) 데이터 셋 불러오기
df = pd.read_csv("data/diabetes.csv")
print(df.shape)
print(df.head())

# 2) 데이터 학습/평가 = 80: 20 로 분리 기준 만들기

split_count = int(df.shape[0] * 0.8)  # 80% 분리
train = df[:split_count].copy()
print(train.shape)
test = df[split_count:].copy()
print(train.shape)

# 3) 학습/예측에 사용할 컬럼 설정
feature_names = train.columns[:-1].tolist()  # 처음부터 마지막 전까지 (마지막은 정답)
print(feature_names)
label_name = train.columns[-1]  # 마지막 컬럼 = 정답 컬럼
print(label_name)

# 4) 학습/예측 데이터 셋 만들기

X_train = train[feature_names]  # 학습에 사용할 컬럼만 선정 , feature_names 변수에 들어있는 열만 가져옵니다.
print(X_train.shape)
print(X_train.head())

y_train = train[label_name]   # 결과에 사용할 컬럼만 선정
print(y_train.shape)
print(y_train.head())

# 5) 예측에 사용할 데이터 셋을 만든다.  Y_test는 답안지로 사용

X_test = test[feature_names]
print(X_test.shape)
print(X_test.head())

y_test = test[label_name]
print(y_test.shape)
y_test.head()

# 6) 머신러닝 알고리즘 (Decision Tree 알고리즘)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()  # DT 알고리즘으로 model을 선정

model = model.fit(X_train, y_train)  # 학습 시킴
# print(model.fit(X_train, y_train))

y_predict = model.predict(X_test) # 성능 예측하기
print(y_predict[:5])


# 7) 예측한 모델의 성능 측정 하기

diff_count = abs(y_test - y_predict).sum()  # 정답과 예측의 차이를 합한다, 정답 = 예측이 같으면 0으로 높을수록 오답임
print(diff_count)

print(abs(y_test - y_predict).sum() / len(y_test))   # 예측과 정답의 차이의 합을 전체로 나눔 = 오분류율
accuracy = round((len(y_test) - diff_count) / len(y_test) * 100,4) # 71% 의 정합률을 보인다
print("정확도 :" + str(accuracy)+"%")

from sklearn.metrics import accuracy_score

accuracy_score = round(accuracy_score(y_test, y_predict) * 100,4)
print("정확도 :" + str(accuracy_score)+"%")

# 8) 예측한 모델의 성능을 tree로 보여주기

from sklearn.tree import plot_tree

# plot_tree(model, feature_names=feature_names)
# plt.figure(figsize=(20, 20))
# tree = plot_tree(model, feature_names=feature_names, filled=True, fontsize=10)
# plt.show()

# import graphviz
# from sklearn.tree import export_graphviz
#
# dot_tree = export_graphviz(model, feature_names = feature_names, filled=True)
# graphviz.Source(dot_tree)
# plt.show()
# print("완료")

# 9) 중요 피쳐정보에 대해서 확인

model_important  = model.feature_importances_
print(model_important)
# model.feature_importances_
sns.barplot(x=model.feature_importances_, y=feature_names)
plt.show()
print("완료")
