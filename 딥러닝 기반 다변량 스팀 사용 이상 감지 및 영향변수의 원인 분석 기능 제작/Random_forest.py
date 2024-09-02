#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader, IterableDataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


df = pd.read_excel('df_ext(2023-04-01~2023-08-31,51250385)_2023-10-17 10-58-30 -seoultec.xlsx')
df = df.reset_index(drop=True)

# 끊긴 지점으로 나눴을 때, 끊긴 부분의 데이터셋 구하고 그 길이가 어떻게 되는지 조사

# 시간 간격의 차이를 계산
df['time_diff'] = df['date'].diff()

# 일정 값(예: 1일)보다 큰 차이를 가지는 위치 확인
non_continuous = df[df['time_diff'] > pd.Timedelta(minutes=1)]
time_gap_list= non_continuous.index.to_list()
time_gap_list.insert(0, 0)
time_gap_list.append(len(df))

# jr 변수 추가
df['date'] = pd.to_datetime(df['date'])
category_start_end_dates = df.groupby('jr')['date'].agg(['min', 'max'])

# 진행 비율을 계산하는 함수
def calculate_progress(row):
    start_date = category_start_end_dates.loc[row['jr'], 'min']
    end_date = category_start_end_dates.loc[row['jr'], 'max']
    current_date = row['date']
    
    # 분모가 0이 되는 경우를 방지합니다.
    if start_date != end_date:
        # total_duration = (end_date - start_date).total_seconds()
        elapsed_time = (current_date - start_date).total_seconds() / 60.0
        progress = elapsed_time 
        return progress
    else:
        # 카테고리 원소가 단 하루만 존재하는 경우 진행률은 1이 됩니다.
        return 1.0

# 'progress' 컬럼을 계산하여 데이터프레임에 추가합니다.
df['jr_progress'] = df.apply(calculate_progress, axis=1)

sub_df_len_list = []
time_date_day = []

for time_gap in range(len(time_gap_list)-1):
    sub_df = df.iloc[time_gap_list[time_gap] : time_gap_list[time_gap+1]]
    sub_df_len_list.append(len(sub_df))
    time_date_day.append(sub_df['date'].iloc[0])
    
min(sub_df_len_list)

# 필요없는 컬럼 삭제
df = df.drop(['time_diff', 'sstable', 'jr', 'output', 'stop', 'shift', 'wclass'], axis=1)

def is_continuous(series):
    return all((series.shift(-1) - series).dropna() == pd.Timedelta(minutes=1))

window_size = 20
time_gap = 10

X = []
y = []

#윈도우 사이즈와 지연 예측 시간 고려하여 시계열 데이터로 변환
for i in range(len(df) - window_size + 1 - time_gap):
    subset = df.iloc[i:i+window_size + time_gap]
    if is_continuous(subset['date']):
        X.append(subset.iloc[:window_size])
        y.append(subset.iloc[-1]['ei']) 

def find_discontinuous_points(series):
    discontinuous_points = []
    for i in range(1, len(series)):
        if series.iloc[i] - series.iloc[i-1] != pd.Timedelta(minutes=1):
            discontinuous_points.append(i)
    return discontinuous_points

# 시간이 끊긴 지점을 기준으로 데이터 나누기
break_points = find_discontinuous_points(pd.concat([x['date'] for x in X]))

# 8:2 비율에 가장 근접한 끊긴 지점 찾기
target_ratio = 0.8
best_point = None
best_ratio = float('inf')
total_length = len(X)
for point in break_points:
    ratio = point / total_length
    if abs(ratio - target_ratio) < best_ratio:
        best_ratio = abs(ratio - target_ratio)
        best_point = point

# 찾은 지점을 기준으로 데이터 나누기
if best_point is not None:
    X_train = X[:best_point]
    y_train = y[:best_point]
    X_test = X[best_point:]
    y_test = y[best_point:]

X_train = [x.drop(columns=['date']) for x in X_train]
X_test = [x.drop(columns=['date']) for x in X_test]

X_train_array = [x.values for x in X_train]
X_test_array = [x.values for x in X_test]

# StandardScaler 학습
scaler = StandardScaler().fit(np.concatenate(X_train_array, axis=0))

# X_train과 X_test 스케일링
X_train_scaled = [scaler.transform(x) for x in X_train_array]
X_test_scaled = [scaler.transform(x) for x in X_test_array]

#시각화를 위해 각 인스턴스에 컬럼 정보 저장

def rolling_window_sequences_and_names(X, window_size, original_columns):
    """
    X : List of numpy arrays, where each numpy array is a sequence.
    window_size : Size of the window to use for rolling.
    original_columns : Original column names to use for renaming.

    Returns:
    X_rolled : List of rolled sequences with new column names
    """
    X_rolled = []
    
    for sequence in X:
        # Rolling the array and renaming columns
        for start in range(0, sequence.shape[0] - window_size + 1):
            window = sequence[start:start+window_size]
            window_flattened = window.flatten()
            new_columns = ['t{}_{}'.format(i, col) for i in range(window_size) for col in original_columns]
            rolled_df = pd.DataFrame([window_flattened], columns=new_columns)
            X_rolled.append(rolled_df)
    
    return pd.concat(X_rolled, axis=0)

original_columns = X_train[0].columns

# Rolling windows for X_train_scaled and X_test_scaled with new column names
X_train_df = rolling_window_sequences_and_names(X_train_scaled, window_size, original_columns)
X_test_df = rolling_window_sequences_and_names(X_test_scaled, window_size, original_columns)

print(X_train_df.shape)



def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print("Folder already exists, no action taken.")

# 예제 사용
folder_path = './machine learning/'
create_folder_if_not_exists(folder_path)


# 랜덤 포레스트 모델 생성 및 학습
rf_regressor = RandomForestRegressor(n_estimators=100)
rf_regressor.fit(X_train_df, y_train)

# 예측
y_pred = rf_regressor.predict(X_test_df)

# 성능 평가: MSE 사용
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error(Randomforest): {mse}")

r2 = r2_score(y_test, y_pred)
print(f"R^2 Score(Randomforest): {r2}")


# 성능 지표를 텍스트 파일로 저장
with open( folder_path + 'model_performance.txt', 'w') as file:
    file.write(f'r2: {r2}\n')
    file.write(f'mse: {mse}\n')

y_test_original = y_test
y_pred_original = y_pred

sns.jointplot(x=y_test_original, y=y_pred_original.flatten(), kind='reg')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.savefig(folder_path + 'Random Forest scatter plot.png')

plt.show()


plt.figure(figsize=(12, 6))
plt.plot(y_test_original, label="Actual Values", color='blue')
plt.plot(y_pred_original.flatten(), label="Predictions", color='red', linestyle='--')
plt.title("Time Series Prediction_Random forest")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.savefig(folder_path + 'Random Forest time series prediction plot.png')
plt.show()


#랜덤 포레스트 변수 중요도 시각화

ftr_importances_values = rf_regressor.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index=X_test_df.columns)
ftr_top = ftr_importances.sort_values(ascending=False)[:20]

plt.figure(figsize=(8, 6))
bars = sns.barplot(x=ftr_top, y=ftr_top.index)

# 각 막대에 중요도 값을 추가
for idx, val in enumerate(ftr_top):
    bars.text(val, idx, 
              f'{val:.4f}', 
              va='center', ha='left', color='black')

plt.tight_layout()
plt.title("Top 20 Feature Importances (Coefficients) in Random forest")
plt.savefig(folder_path + 'Random Forest feature plot.png')
plt.show()

