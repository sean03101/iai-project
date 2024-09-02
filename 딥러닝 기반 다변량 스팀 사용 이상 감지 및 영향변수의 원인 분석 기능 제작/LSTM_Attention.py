import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset, DataLoader, IterableDataset

import shap

from sklearn.preprocessing import StandardScaler, MinMaxScaler



#데이터로드
df = pd.read_excel('df_ext(2023-04-01~2023-08-31,51250385)_2023-10-17 10-58-30 -seoultec.xlsx')
df.tail()
df = df.reset_index(drop=True)

# jr의 특정값이 나타난 후에 사라졌다가 다시 나타나는 패턴이 있는지 체크
df['jr'] = df['jr'].astype(str)
categories = df['jr'].unique()
pattern_exists = {cat: False for cat in categories}

#각 카테고리에 대해 체크
for cat in categories:
    # 카테고리가 변경되는 부분을 찾음
    shifts = df['jr'] != df['jr'].shift()
    # 첫 번째 원소가 변경 지점으로 간주되지 않도록 함
    shifts.iloc[0] = False
    # 각 카테고리가 변경된 지점에 대한 인덱스를 가져옴
    cat_shifts = df[shifts & (df['jr'] == cat)].index
    
    # 현재 카테고리가 cat인지 확인하고 그 인덱스를 가져옴
    cat_indices = df[df['jr'] == cat].index
    
    # 각 변경 지점에 대해, 그 이전에 다른 카테고리가 있었는지 확인
    for shift_index in cat_shifts:
        # 변경 지점 이전에 다른 카테고리가 있는지 확인
        previous_indices = cat_indices[cat_indices < shift_index]
        # 변경 지점 이후에 해당 카테고리가 다시 나타나는지 확인
        following_indices = cat_indices[cat_indices > shift_index]
        
        # 이전과 이후에 해당 카테고리 인덱스가 존재하는 경우, 패턴이 존재함
        if previous_indices.any() and following_indices.any():
            pattern_exists[cat] = True
            break  # 패턴을 찾았으니 더 이상 검사할 필요 없음

# 결과 출력
true_patterns = {k: v for k, v in pattern_exists.items() if v}
print(true_patterns)

# jr 변수 추가
df['date'] = pd.to_datetime(df['date'])
category_start_end_dates = df.groupby('jr')['date'].agg(['min', 'max'])

# 진행 비율을 계산하는 함수
def calculate_progress(row):
    start_date = category_start_end_dates.loc[row['jr'], 'min']
    end_date = category_start_end_dates.loc[row['jr'], 'max']
    current_date = row['date']
    
    # 분모가 0이 되는 경우를 방지
    if start_date != end_date:
        # total_duration = (end_date - start_date).total_seconds()
        elapsed_time = (current_date - start_date).total_seconds() / 60.0
        progress = elapsed_time 
        return progress
    else:
        # 카테고리 원소가 단 하루만 존재하는 경우 진행률은 1
        return 1.0

# 'progress' 컬럼을 계산하여 데이터프레임에 추가
df['jr_progress'] = df.apply(calculate_progress, axis=1)


# 각 'jr' 카테고리 내에서 이전 시간과의 차이 계산
df['time_diff2'] = df.groupby('jr')['date'].diff()

# 1분 초과 시간 차이가 나는 레코드 찾기
time_gaps = df[df['time_diff2'] > pd.Timedelta(minutes=1)]

# 해당 'jr' 카테고리만 출력
categories_with_gaps = time_gaps['jr'].unique()

# 유니크한 카테고리의 개수 출력
print((categories_with_gaps))


# 시간 간격의 차이를 계산
df['time_diff'] = df['date'].diff()

# 일정 값(예: 1일)보다 큰 차이를 가지는 위치 확인
non_continuous = df[df['time_diff'] > pd.Timedelta(minutes=1)]
time_gap_list= non_continuous.index.to_list()
time_gap_list.insert(0, 0)
time_gap_list.append(len(df))


sub_df_len_list = []
time_date_day = []

for time_gap in range(len(time_gap_list)-1):
    sub_df = df.iloc[time_gap_list[time_gap] : time_gap_list[time_gap+1]]
    sub_df_len_list.append(len(sub_df))
    time_date_day.append(sub_df['date'].iloc[0])
    
min(sub_df_len_list)

# 필요없는 컬럼 삭제
df = df.drop(['time_diff', 'time_diff2', 'sstable', 'jr', 'output', 'stop', 'shift', 'wclass'], axis=1)

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

# 60%에 가장 근접한 끊긴 지점 찾기
val_ratio = 0.6
best_point = None
best_ratio = float('inf')
total_length = len(X)
for point in break_points:
    ratio = point / total_length
    if abs(ratio - val_ratio) < best_ratio:
        best_ratio = abs(ratio - val_ratio)
        val_best_point = point

# 80%에 가장 근접한 끊긴 지점 찾기
best_point = None
best_ratio = float('inf')
total_length = len(X)
test_ratio = 0.8
for point in break_points:
    ratio = point / total_length
    if abs(ratio - test_ratio) < best_ratio:
        best_ratio = abs(ratio - test_ratio)
        test_best_point = point


# 찾은 지점을 기준으로 데이터 나누기
if val_ratio is not None:
    X_train = X[:val_best_point]
    y_train = y[:val_best_point]
    X_val = X[val_best_point:test_best_point]
    y_val = y[val_best_point:test_best_point]
    X_test = X[test_best_point:]
    y_test = y[test_best_point:]


X_train = [x.drop(columns=['date']) for x in X_train]
X_val = [x.drop(columns=['date']) for x in X_val]
X_test = [x.drop(columns=['date']) for x in X_test]

X_train_array = [x.values for x in X_train]
X_val_array = [x.values for x in X_val]
X_test_array = [x.values for x in X_test]

# StandardScaler 학습
scaler = MinMaxScaler().fit(np.concatenate(X_train_array, axis=0))

# X_train과 X_test 스케일링
X_train_scaled = [scaler.transform(x) for x in X_train_array]
X_val_scaled = [scaler.transform(x) for x in X_val_array]
X_test_scaled = [scaler.transform(x) for x in X_test_array]
print(np.array(X_test_scaled).shape)


#numpy to pytorch tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)  
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)  
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)  

# TensorDataset 생성
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor) 

print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))

#Tensor Dataloader 생성
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=6341, shuffle=False)

dataiter = iter(train_dataloader)
x_sample, y_sample = dataiter.next()

print(x_sample.shape)  
print(y_sample.shape)

# 어텐션 스코어 추출을 위한 추가적인 어텐션 레이어
class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.attention_weight = nn.Parameter(torch.FloatTensor(feature_dim, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_in):
        attention_probs = self.softmax(torch.matmul(x_in, self.attention_weight).squeeze(2))
        weighted_sequence = torch.bmm(x_in.permute(0, 2, 1), attention_probs.unsqueeze(2)).squeeze(2)
        return weighted_sequence, attention_probs   # 시각화를 위해 어텐션 스코어와 weight 동시에 리턴

# 단순 LSTM 코드
class ComplexLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.5):
        super(ComplexLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout_prob, batch_first=True)
        self.attention = Attention(hidden_dim)

        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out, attention_probs = self.attention(out)

        out = self.batch_norm(out)
        out = self.dropout(out)

        out = torch.tanh(self.fc1(out))
        out = torch.tanh(self.fc2(out))
        out = self.fc3(out)

        return out, attention_probs

#하이퍼파리미터
input_dim = 40
hidden_dim = 256
num_layers = 4
output_dim = 1
dropout_prob = 0.4



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ComplexLSTM(input_dim, hidden_dim, num_layers, output_dim, dropout_prob).to(device)

criterion = nn.L1Loss()

num_epochs = 100
train_losses = []
val_losses = []

optimizer = optim.Adam(model.parameters(), lr=1e-4)

import os
# 폴더 생성
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print("Folder already exists, no action taken.")

folder_path = './LSTM/'
create_folder_if_not_exists(folder_path)

best_val_loss = float('inf')  # 초기에는 무한대로 설정
model_save_path = 'LSTM/best_model.pth'  # 모델을 저장할 경로

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        outputs, _ = model(X_batch)
        
        loss = criterion(outputs.squeeze(), y_batch.squeeze())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)
    
    train_loss /= len(train_dataloader.dataset)
    train_losses.append(train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_val, y_val in val_dataloader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            
            outputs, _ = model(X_val)
            loss = criterion(outputs.squeeze(), y_val.squeeze())
            
            val_loss += loss.item() * X_val.size(0)
            
        val_loss /= len(val_dataloader.dataset)
        val_losses.append(val_loss)
    
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Best model 저장
    if val_loss < best_val_loss:
        print(f"Validation Loss Improved ({best_val_loss:.6f} -> {val_loss:.6f}). Saving model...")
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_save_path)
        
torch.save(model.state_dict(), './LSTM/last_model.pth')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ComplexLSTM(input_dim, hidden_dim, num_layers, output_dim, dropout_prob).to(device)
model_save_path = 'LSTM/best_model.pth'  # 모델을 저장했던 경로


# Loss 시각화
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title("Train and Validation Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(folder_path+'loss.png',dpi=300)
plt.clf()



# 모델을 평가 모드로 설정
model = ComplexLSTM(input_dim, hidden_dim, num_layers, output_dim, dropout_prob).to(device)
model.load_state_dict(torch.load(model_save_path))
model.eval()
all_predictions = []
all_labels = []
all_attention_maps = []

#테스트 단계
model.eval()
with torch.no_grad():
    for X_batch, y_batch in test_dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs, attention_map = model(X_batch)
        
        all_predictions.append(outputs.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())
        all_attention_maps.append(attention_map.cpu().numpy())
        
        
        
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Convert to numpy arrays
all_predictions = np.concatenate(all_predictions) 
all_labels = np.concatenate(all_labels)

# Calculate metrics
mse = mean_squared_error(all_labels, all_predictions)
mae = mean_absolute_error(all_labels, all_predictions)
r2 = r2_score(all_labels, all_predictions)

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")


# 성능 지표를 텍스트 파일로 저장
with open(folder_path+'model_performance.txt', 'w') as file:
    file.write(f'MSE: {mse}\n')
    file.write(f'R2 Score: {r2}\n')


#모든 테스트 데이터셋에 대한 어텐션 웨이트 추출 후 평균내어 시각화
all_attention_maps = np.concatenate(all_attention_maps, axis=0)
avg_attention_weights = all_attention_maps.mean(axis=0)

plt.figure(figsize=(15, 5))
plt.plot(avg_attention_weights)
plt.xlabel("Time Steps")
plt.ylabel("Average Attention Weight")
plt.title("Average Attention Map across all test instances")
plt.savefig(folder_path+'Average Attention Map.png',dpi=300)
plt.clf()


plt.figure(figsize=(12, 6))
plt.plot(all_labels, label="Actual Values", color='blue')
plt.plot(all_predictions, label="Predictions", color='red', linestyle='--')
plt.title("Time Series Prediction_LSTM")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.savefig(folder_path+'Time Series Prediction.png',dpi=300)
plt.clf()




sns.jointplot(x=all_labels, y=all_predictions.flatten(), kind='reg')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.savefig(folder_path+'Scatter plot.png',dpi=300)
plt.clf()


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # 모델이 출력과 숨겨진 상태를 반환한다고 가정하고, 출력만 사용
        output, _ = self.model(x)
        return output
    
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ComplexLSTM(input_dim, hidden_dim, num_layers, output_dim, dropout_prob).to(device)
model.load_state_dict(torch.load(model_save_path))
model_wrapper = ModelWrapper(model).to(device)

# SHAP 초기화
background_data = torch.tensor(X_train_scaled[:3000], dtype=torch.float32).to(device)
explainer = shap.DeepExplainer(model_wrapper, background_data)

# 테스트 데이터 준비
all_test_data = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

# 모델 평가 모드 설정
model.eval()
torch.backends.cudnn.enabled = False
# SHAP 값 계산
shap_values = explainer.shap_values(all_test_data)

torch.backends.cudnn.enabled = True

# shap_values가 리스트인 경우 numpy 배열로 변환
if isinstance(shap_values, list):
    shap_values = np.array(shap_values[0])

# 시간 단계별로 평균내는 과정
average_shap_values_over_time = np.mean(shap_values, axis=1)

# 평균낸 SHAP 값의 형태를 (샘플 수, 특성 수)로 변경
shap_values_averaged = average_shap_values_over_time.reshape(shap_values.shape[0], -1)

# 테스트 데이터를 시간 단계별로 평균냄
all_test_data_averaged = np.mean(all_test_data.cpu().numpy(), axis=1)

# SHAP 값과 테스트 데이터의 형태가 일치하는지 확인
assert shap_values_averaged.shape == all_test_data_averaged.shape

# 시각화
feature_names = X_train[0].columns.tolist()  # Pandas DataFrame의 열 이름 사용
shap.summary_plot(shap_values_averaged, all_test_data_averaged, feature_names=feature_names)
plt.savefig(folder_path+'SHAP.png',dpi=300)
plt.clf()
# In[27]:


# 예를 들어, shap_values가 (samples, outputs, features)의 3차원 배열이라고 가정
# 여기서는 모든 출력에 대해 평균을 내고 싶은 경우
mean_abs_shap_values = np.mean(np.abs(shap_values), axis=(0, 1))


# 특성 이름
feature_names = X_train[0].columns.tolist()

# 특성 중요도와 이름을 DataFrame으로 변환
feature_importances = pd.DataFrame({
    'feature': feature_names,
    'importance': mean_abs_shap_values
})

# 중요도에 따라 특성을 정렬
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

# 중요도에 따라 특성을 정렬하고 상위 10개를 선택
top_feature_importances = feature_importances.sort_values(by='importance', ascending=False).head(10)

# 바 차트로 상위 10개 특성 중요도를 시각화
top_feature_importances.plot(kind='bar', x='feature', y='importance', legend=False)
plt.title('Top 10 Feature Importances using SHAP')
plt.ylabel('Average Impact on Model Output')
plt.xticks(rotation=45)  # 특성 이름이 길 경우 회전시켜서 라벨이 겹치지 않도록 설정
plt.savefig(folder_path+'SHAP cumulative.png',dpi=300)
plt.clf()