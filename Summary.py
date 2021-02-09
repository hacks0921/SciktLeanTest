import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import pydot


# 1)데이터 불러오기
df = pd.read_csv("data/diabetes2.csv", encoding='CP949')
print(df.shape)

# 2)X 컬럼 선정하기
print(df.columns)
X = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI','DiabetesPedigreeFunction', 'Age']]
print(X.shape)
# 2)Y 컬럼 선정하기
y = df['Outcome']
print(y.shape)

##################################################################

feature_columns = df.columns[:-1].tolist()              # df에서 feature가 있는 컬럼의 이름을 추출
X = df[feature_columns]                                 # feature가 있는 컬럼만 X라는 dataframe으로 만들어줌
answer_columns = df.columns[-1]                         # df에서 정답이적혀있는 컬럼이름을 추출(마지막 열 선택(-1))
y = df[answer_columns]                                  # 정답 컬럼만 따로 y라는 dataframe으로 만들어줌

class_names_values = df[answer_columns].value_counts()  # 정답 클레스 유형 (OK:00개, NG:00개 형식으로 추출)
class_names =class_names_values.keys().tolist()         # Class_nmaes_values에서 Key 값을가지고 List 형태로 변경함
print(class_names)
##################################################################


# 3) X,y Train / X,y Test Set 분류
from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train)
print(X_test)
print(y_train)
print(y_test)



# 4) 모델링 선택
from sklearn.tree import DecisionTreeClassifier   # Decision Tree
model_DT = DecisionTreeClassifier(random_state=42,
                                  max_depth= 5)
#
from sklearn.ensemble import RandomForestClassifier   # Random Forest
model_RF =RandomForestClassifier(random_state=42)
#
from sklearn.ensemble import GradientBoostingClassifier   # Gradi
model_GB = GradientBoostingClassifier(random_state=42)
#
# # 5) 모델 학습
model_DT.fit(X_train, y_train)
model_RF.fit(X_train, y_train)
model_GB.fit(X_train, y_train)

# # 6) 예측
y_predict_DT = model_DT.predict(X_test)
y_predict_RF = model_RF.predict(X_test)
y_predict_GB = model_GB.predict(X_test)


# result_Train = pd.concate([X_train, y_train])
print("*"*100)
print(X_train)
print(y_train)
# print(y_predict_DT)
y_predict_DT = pd.DataFrame(y_predict_DT, columns=["y_predict_DT"])
y_predict_RF = pd.DataFrame(y_predict_RF, columns=["y_predict_RF"])
y_predict_GB = pd.DataFrame(y_predict_GB, columns=["y_predict_GB"])

print(y_predict_DT)
result_Test = pd.concat([X_test, y_test], axis=1 , ignore_index= False)
result_Test = result_Test.reset_index() # 인덱스 초기화
result_Test["y_predict_DT"] = y_predict_DT  # result_Test 데이터 프레임에 예상 결과 붙이기
result_Test["y_predict_RF"] = y_predict_RF  # result_Test 데이터 프레임에 예상 결과 붙이기
result_Test["y_predict_GB"] = y_predict_GB  # result_Test 데이터 프레임에 예상 결과 붙이기


print(result_Test)
result_Test.to_csv("result_test.csv")

# print(y_predict_DT)
# df["y_predict_DT"] = df[y_predict_DT]
#
# # y_predict[:5]
# print((y_predict_DT != y_test).sum())  # 정답과 다른 애들의 합계
# print((y_predict_RF != y_test).sum())  # 정답과 다른 애들의 합계
# print((y_predict_GB != y_test).sum())  # 정답과 다른 애들의 합계
#
from sklearn.metrics import accuracy_score
print("y_predict_DT 정합성 :", accuracy_score(y_test, y_predict_DT))
print("y_predict_RF 정합성 :", accuracy_score(y_test, y_predict_RF))
print("y_predict_GB 정합성 :", accuracy_score(y_test, y_predict_GB))

# print(accuracy_score(y_test, y_predict_RF))
# print(accuracy_score(y_test, y_predict_GB))

# 7) 중요한 Feature 선택
# feature_names = X_train.columns.tolist()
# sns.barplot(x=model_DT.feature_importances_, y=feature_names)
# plt.show()

# 8) 의사결정 나무 그림그리기

from sklearn.tree import export_graphviz
export_graphviz(model_DT, out_file='tree.dot',
                class_names = class_names,
                feature_names= feature_columns,
                impurity=True,
                filled=True)
(graph,) = pydot.graph_from_dot_file('tree.dot', encoding='utf8')
graph.write_png('tree_picture.png')