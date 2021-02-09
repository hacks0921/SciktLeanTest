import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import pydot

# 1)데이터 불러오기
df = pd.read_csv("data/diabetes2.csv", encoding='CP949')
print(df.shape)

##################################################################
# 2)X 컬럼 선정하기
# print(df.columns)
# X = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI','DiabetesPedigreeFunction', 'Age']]
# print(X.shape)
# # 2)Y 컬럼 선정하기
# y = df['Outcome']
# print(y.shape)

feature_columns = df.columns[:-1].tolist()              # df에서 feature가 있는 컬럼의 이름을 추출
X = df[feature_columns]                                 # feature 있는 컬럼만 X라는 dataframe으로 만들어줌
answer_columns = df.columns[-1]                         # df에서 정답이적혀있는 컬럼이름을 추출(마지막 열 선택(-1))
y = df[answer_columns]                                  # 정답 컬럼만 따로 y라는 dataframe으로 만들어줌

class_names_values = df[answer_columns].value_counts()  # 정답 클레스 유형 (OK:00개, NG:00개 형식으로 추출)
class_names =class_names_values.keys().tolist()         # Class_nmaes_values에서 Key 값을가지고 List 형태로 변경함
print(class_names)

##################################################################

## Standard  Scalling  # standar Scalling
from sklearn.preprocessing import StandardScaler            # 평균 0 , 표준편차 1
sdscaler = StandardScaler()
X = sdscaler.fit_transform(X)
X  = pd.DataFrame(X, columns=feature_columns)
print(X)

##################################################################

# 3) X,y Train / X,y Test Set 분류
from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# 4) 모델링 선택
from sklearn.tree import DecisionTreeClassifier   # Decision Tree
model_DT = DecisionTreeClassifier(random_state=42,max_depth= 5)
# #
from sklearn.ensemble import RandomForestClassifier   # Random Forest
model_RF =RandomForestClassifier(random_state=42)
# #
from sklearn.ensemble import GradientBoostingClassifier   # Gradi
model_GB = GradientBoostingClassifier(random_state=42)
#
# # 5) 모델 학습
model_DT.fit(X_train, y_train)
model_RF.fit(X_train, y_train)
model_GB.fit(X_train, y_train)

# # 6) 예측
y_predict_DT = model_DT.predict(X_test)             # X_test data를 가지고 결과 예측(DT)
y_predict_RF = model_RF.predict(X_test)             # X_test data를 가지고 결과 예측(RF)
y_predict_GB = model_GB.predict(X_test)             # X_test data를 가지고 결과 예측(GB)

print((y_predict_DT != y_test).sum())               # 정답과 다른 애들의 합계
print((y_predict_RF != y_test).sum())               # 정답과 다른 애들의 합계
print((y_predict_GB != y_test).sum())               # 정답과 다른 애들의 합계

result_Test0 = pd.concat([X_test, y_test], axis=1 , ignore_index= False)  # test data set  합치기
result_Test0 = result_Test0.reset_index()                                  # 인덱스 초기화

result_Test0["y_predict_DT"] = y_predict_DT  # result_Test 데이터 프레임에 예상 결과 붙이기
result_Test0["y_predict_RF"] = y_predict_RF  # result_Test 데이터 프레임에 예상 결과 붙이기
result_Test0["y_predict_GB"] = y_predict_GB  # result_Test 데이터 프레임에 예상 결과 붙이기

result_Test0.to_csv("result_Test0.csv")        # result_raw에 결과 저장

from sklearn.metrics import accuracy_score
print("y_predict_DT 정합성(test_기본) :", accuracy_score(y_test, y_predict_DT))
print("y_predict_RF 정합성(test_기본) :", accuracy_score(y_test, y_predict_RF))
print("y_predict_GB 정합성(test_기본) :", accuracy_score(y_test, y_predict_GB))

# 7) 중요한 Feature 선택
feature_names = X_train.columns.tolist()
sns.barplot(x=model_DT.feature_importances_, y=feature_names)
plt.savefig('feature_importances.png')
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

#######################################################################################################################

# 변수 설정하기 (max_depth, max_features)
from sklearn.model_selection import RandomizedSearchCV

max_depth = np.random.randint(2, 20, 10) #2부터 20까지 10개 만든다
max_features = np.random.uniform(0.3, 1.0, 10)  # 0.3~1까지 Feature를 일부만 가지고와서 쓸떄 10개
param_distributions = {"max_depth": max_depth,"max_features": max_features}

# 여러개 모델 동시에 학습/측정 (estimator = 추정)
estimators = [DecisionTreeClassifier(random_state=42),
             RandomForestClassifier(random_state=42),  # 병렬 처리 학습 (Bagging)
             GradientBoostingClassifier(random_state=42)]  # 틀린 문제에 가중을 주기 때문에 순차 학습으로 시간이 오래걸림 (Boosting)

results = []
for estimator in estimators:
    result = []
    print(estimator.__class__.__name__)  #이름만 출력한다
    if estimator.__class__.__name__ != "DecisionTreeClassifier":
        param_distributions["n_estimators"] = np.random.randint(100,1000,10) # DT가 아닐경우 n_estimator를 추가
        #n_estimators : randoem forest에서 몇번의 DT를 할것인지 사용 --> 증가할수록 알고리즘 오래 걸림
    clf = RandomizedSearchCV(estimator, param_distributions,
                             n_iter=5,  # 100번 이터레이션
                             scoring="accuracy",  # 정확도를 스코어로 설정
                             n_jobs=-1,  # 사용할수있는 CPU자원 몇개 사용?
                             cv=5,  # Cross validation 조각 / 모의고사 풀어볼떄 몇회??
                             verbose=1  # 로그를 찍을지 말지 0: 안찍음 , 1,2: 찍는다
                             )

    if estimator.__class__.__name__ == "DecisionTreeClassifier":
        print("DecisionTreeClassifier 모델")
        model_DT = clf.fit(X_train, y_train)
        y_predict_DT = model_DT.predict(X_test)
    elif estimator.__class__.__name__ == "RandomForestClassifier":
        print("RandomForestClassifier 모델")
        model_RF = clf.fit(X_train, y_train)
        y_predict_RF = model_RF.predict(X_test)
    elif estimator.__class__.__name__ == "GradientBoostingClassifier":
        print("GradientBoostingClassifier 모델")
        model_RF = clf.fit(X_train, y_train)
        y_predict_GB = model_RF.predict(X_test)

    result.append(estimator.__class__.__name__) # 알고리즘 이름
    result.append(clf.best_params_)  # 파라미터
    result.append(clf.best_score_)   # 최고 스코어
    result.append(clf.score(X_test, y_test))
    result.append(clf.cv_results_)   # 결과
    results.append(result) # Results에 결과 누적

df = pd.DataFrame(results, columns= ["estimator", "best_params", "train_score", "test_score", "cv_result"])
df.to_csv("Results_BestModel.csv")

#######################################################################################################################

# result_Train = pd.concate([X_train, y_train])
print("*"*100)
print(X_train)
print(y_train)
# print(y_predict_DT)
y_predict_DT = pd.DataFrame(y_predict_DT, columns=["y_predict_DT"])
y_predict_RF = pd.DataFrame(y_predict_RF, columns=["y_predict_RF"])
y_predict_GB = pd.DataFrame(y_predict_GB, columns=["y_predict_GB"])

print(y_predict_DT)
result_Test = pd.concat([X_test, y_test], axis=1 , ignore_index= False)  # test data set  합치기
result_Test = result_Test.reset_index()                                  # 인덱스 초기화

result_Test["y_predict_DT"] = y_predict_DT  # result_Test 데이터 프레임에 예상 결과 붙이기
result_Test["y_predict_RF"] = y_predict_RF  # result_Test 데이터 프레임에 예상 결과 붙이기
result_Test["y_predict_GB"] = y_predict_GB  # result_Test 데이터 프레임에 예상 결과 붙이기

print(result_Test)
result_Test.to_csv("result_Test1.csv")        # result_raw에 결과 저장

from sklearn.metrics import accuracy_score
print("y_predict_DT 정합성(test) :", accuracy_score(y_test, y_predict_DT))
print("y_predict_RF 정합성(test) :", accuracy_score(y_test, y_predict_RF))
print("y_predict_GB 정합성(test) :", accuracy_score(y_test, y_predict_GB))