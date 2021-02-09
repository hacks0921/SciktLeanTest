import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# 2.1.1 당뇨병 데이터셋 미리보기
df = pd.read_csv("data/diabetes.csv")
print(df.shape)
print(df.head())
print(df.info())

 # 결측치 데이터 보기
df_null = df.isnull()
print(df_null.head())
print(df_null.sum()) # 결측치 데이터 개수 확인
print(df.describe()) # 각 feature의 특징 확인
df2 = df.describe()

# writedata.py
df2.to_csv('df2.csv')  # 정보를 csv file로 저장 하기

# 처음 시작열부터 마지막에서 1열 전까지 가지고와서 list로 만들어준다
feature_columns = df.columns[0:-1].tolist()
print(feature_columns)
print(df.columns)
print(df.columns[:]) # 인덱스 형식으로 tolist를 이용해서 list로 바뀌줘야 함

# 2.1.2 결측치 보기

cols = feature_columns[1:] # feature_columns 중에서 1번째 열을 제외하고 나머지를 cols 변수로 가지고온다
print(cols)

df_null = df[cols].replace(0, np.nan)  # 0으로 기록된 값을 null로 변경하고, 결측치를 알아본다
print(df_null)
df_null = df_null.isnull()  # isnull 과 아닌것을 True / False로 분리 하게된다
print(df_null)
print(df_null.sum())  # True인 값들만 합계를 구해서 보여준다


# figure = plt.figure(figsize=(10, 4))
# df_null.sum().plot.barh()
# plt.show()
#
# plt.figure(figsize=(15, 4))
# sns.heatmap(df_null, cmap="Greys_r")  # 결측치를 heat map으로 그려서 본다 True : 1, False : 0
# plt.show()

# 2.1.3 훈련과 예측에 사용할 정답값을 시각화로 보기

print(df["Outcome"])  # 정답값 컬럼을 넣고 확인
print(df["Outcome"].value_counts())  # 정답값의 개수를 확인 한다 0,1이 몇개 씩 있는지 확인 (1은 발병, 0은 발병하지 않는 케이스)
print(df["Outcome"].value_counts(normalize=True))  # 전체 비율로 확인


print(df.groupby(["Pregnancies"])["Outcome"].mean()) # groupby 할때는 index에 올 값을 () 넣고 그 뒤에 [] value를 넣는다
 # 임신(Pregnancies) 횟수가 늘어날수록 발병 횟수가 높음

print(df.groupby(["Pregnancies"])["Outcome"].agg(["mean", "count","std"]))

df_po = df.groupby(["Pregnancies"])["Outcome"].agg(["mean", "count"]).reset_index()
print(df_po)
# df_po["mean"].plot()
# df_po["mean"].plot(kind = 'bar', rot=30)  # BAR Chart 형식으로 그리기, 30도 회전

# sns.countplot(data=df, x="Outcome")  # 발병 횟수를 카운트하여 시각화하기
# sns.countplot(data=df, x="Pregnancies")   #임신횟수 카운트하여 시각화 하기
# sns.countplot(data=df, x="Pregnancies", hue="Outcome")  # 범주 hue를 발병으로 나눠서

df["Pregnancies_high"] = df["Pregnancies"] > 6    #6번 이상 임신하면 True , 6미만 이면 False
print(df[["Pregnancies", "Pregnancies_high"]].head()) # 상위 5개 보여줌
# sns.countplot(data=df, x="Pregnancies_high")
sns.countplot(data=df, x="Pregnancies_high", hue="Outcome")
# 2개의 카테고리(범주)로 나뉜 임신횟수를 카운트해봅니다. 임신횟수가 적은 사람의 수가 더 많습니다.
# 
# df_po.plot()
plt.show()