import autosklearn.classification
import pandas as pd


# 1)데이터 불러오기
df = pd.read_csv("data/diabetes2.csv", encoding='CP949')
print(df.shape)

# 2)X, y 컬럼 선정하기

##################################################################
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
X = pd.DataFrame(X, columns=feature_columns)
##################################################################
# 3) X,y Train / X,y Test Set 분류
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4) Auto ML 평가

automl  = autosklearn.classification.AutoSklearnClassifier()
automl.fit(X_train, y_train)
predictions = automl.predict(X_test)

from sklearn.metrics import accuracy_score
print("AutoML 정합성:", accuracy_score(y_test, predictions))