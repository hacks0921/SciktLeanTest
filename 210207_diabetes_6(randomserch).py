import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1)데이터 불러오기
df = pd.read_csv("data/diabetes_feature.csv")
print(df.shape)



## Standard  Scalling
from sklearn.preprocessing import StandardScaler            # 평균 0 , 표준편차 1
sdscaler = StandardScaler()
X_sdscaler_train = sdscaler.fit_transform(df[:,:-1])
X_sdscaler_train  = pd.DataFrame(X_sdscaler_train, columns=df.columns)
print(X_sdscaler_train)


# 2)X 컬럼 선정하기
print(df.columns)
# X = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Pregnancies_high', 'Age_low', 'Age_middle', 'Insulin_nan', 'low_glu_insulin']]
X = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI','DiabetesPedigreeFunction', 'Age','Pregnancies_high']]
print(X.shape)

# 2)Y 컬럼 선정하기
y = df['Outcome']
print(y.shape)

# 3) X,y Train / X,y Test Set 분류
from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4) 모델링 선택
from sklearn.tree import DecisionTreeClassifier   # Decision Tree
# model_DT = DecisionTreeClassifier(random_state=42)
#
from sklearn.ensemble import RandomForestClassifier   # Random Forest
# model_RF =RandomForestClassifier(random_state=42)
#
from sklearn.ensemble import GradientBoostingClassifier   # Gradi
# model_GB = GradientBoostingClassifier(random_state=42)
#
# # 5) 모델 학습
# model_DT.fit(X_train, y_train)
# model_RF.fit(X_train, y_train)
# model_GB.fit(X_train, y_train)
#
# # 6) 예측
# y_predict_DT = model_DT.predict(X_test)
# y_predict_RF = model_RF.predict(X_test)
# y_predict_GB = model_GB.predict(X_test)
#
# # y_predict[:5]
# print((y_predict_DT != y_test).sum())  # 정답과 다른 애들의 합계
# print((y_predict_RF != y_test).sum())  # 정답과 다른 애들의 합계
# print((y_predict_GB != y_test).sum())  # 정답과 다른 애들의 합계
#
# from sklearn.metrics import accuracy_score
# print(accuracy_score(y_test, y_predict_DT))
# print(accuracy_score(y_test, y_predict_RF))
# print(accuracy_score(y_test, y_predict_GB))

# # 7) 중요한 Feature 선택
# feature_names = X_train.columns.tolist()
# sns.barplot(x=model.feature_importances_, y=feature_names)
# plt.show()

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier   # Decision Tree

# 변수 설정하기 (max_depth, max_features)
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
    clf.fit(X_train, y_train)
    result.append(estimator.__class__.__name__) # 알고리즘 이름
    result.append(clf.best_params_)  # 파라미터
    result.append(clf.best_score_)   # 최고 스코어
    result.append(clf.score(X_test, y_test))
    result.append(clf.cv_results_)   # 결과
    results.append(result) # Results에 결과 누적

df = pd.DataFrame(results, columns= ["estimator", "best_params", "train_score", "test_score", "cv_result"])
df.to_csv("results.csv")

# param_distributions = {"max_depth": max_depth,"max_features": max_features}
# clf = RandomizedSearchCV(estimator,param_distributions,
#                          n_iter=100,  # 100번 이터레이션
#                          scoring="accuracy",  # 정확도를 스코어로 설정
#                          n_jobs=-1,  # 사용할수있는 CPU자원 몇개 사용?
#                          cv=5, # Cross validation 조각 / 모의고사 풀어볼떄 몇회??
#                          verbose= 2 # 로그를 찍을지 말지 0: 안찍음 , 1,2: 찍는다
#                          )
# clf.fit(X_train, y_train)
# print(clf.fit(X_train, y_train))
# print(clf.best_params_)
# print(clf.best_score_)
