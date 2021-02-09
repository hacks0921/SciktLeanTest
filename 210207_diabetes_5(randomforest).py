import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1)데이터 불러오기
df = pd.read_csv("data/diabetes_feature.csv")
print(df.shape)

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
model_DT = DecisionTreeClassifier(random_state=42)

from sklearn.ensemble import RandomForestClassifier   # Random Forest
model_RF =RandomForestClassifier(random_state=42)

from sklearn.ensemble import GradientBoostingClassifier   # Gradi
model_GB = GradientBoostingClassifier(random_state=42)

# 5) 모델 학습
model_DT.fit(X_train, y_train)
model_RF.fit(X_train, y_train)
model_GB.fit(X_train, y_train)

# 6) 예측
y_predict_DT = model_DT.predict(X_test)
y_predict_RF = model_RF.predict(X_test)
y_predict_GB = model_GB.predict(X_test)

# y_predict[:5]
print((y_predict_DT != y_test).sum())  # 정답과 다른 애들의 합계
print((y_predict_RF != y_test).sum())  # 정답과 다른 애들의 합계
print((y_predict_GB != y_test).sum())  # 정답과 다른 애들의 합계

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_predict_DT))
print(accuracy_score(y_test, y_predict_RF))
print(accuracy_score(y_test, y_predict_GB))

# # 7) 중요한 Feature 선택
# feature_names = X_train.columns.tolist()
# sns.barplot(x=model.feature_importances_, y=feature_names)
# plt.show()