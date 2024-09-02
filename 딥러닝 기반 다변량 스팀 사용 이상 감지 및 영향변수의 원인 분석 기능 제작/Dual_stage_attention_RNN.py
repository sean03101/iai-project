import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = pd.read_excel('df_ext(2023-04-01~2023-08-31,51250385)_2023-10-17 10-58-30 -seoultec.xlsx')
df = df.reset_index(drop=True)

df['date'] = pd.to_datetime(df['date'])
category_start_end_dates = df.groupby('jr')['date'].agg(['min', 'max'])

# 진행 비율을 계산하는 함수
def calculate_progress(row):
    start_date = category_start_end_dates.loc[row['jr'], 'min']
    end_date = category_start_end_dates.loc[row['jr'], 'max']
    current_date = row['date']
    
    # 분모가 0이 되는 경우를 방지
    if start_date != end_date:
        total_duration = (end_date - start_date).total_seconds()
        elapsed_time = (current_date - start_date).total_seconds()
        progress = elapsed_time
        return progress
    else:
        # 카테고리 원소가 단 하루만 존재하는 경우 진행률은 1
        return 1.0

# 'progress' 컬럼을 계산하여 데이터프레임에 추가
df['jr_progress'] = df.apply(calculate_progress, axis=1)


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

# 필요없는 컬럼 삭제
df = df.drop(['time_diff', 'sstable', 'jr', 'output', 'stop', 'shift', 'wclass'], axis=1)

def is_continuous(series):
    return all((series.shift(-1) - series).dropna() == pd.Timedelta(minutes=1))

window_size = 20
time_gap = 10

X = []
y = []
dec = []

#윈도우 사이즈와 지연 예측 시간 고려하여 시계열 데이터로 변환
for i in range(len(df) - window_size + 1 - time_gap):
    subset = df.iloc[i:i+window_size + time_gap]
    if is_continuous(subset['date']):
        X.append(subset.iloc[:window_size].drop(columns=['ei']))
        y.append(subset.iloc[-1]['ei'])
        dec.append(subset.iloc[:window_size-1]['ei'])
        
        
def find_discontinuous_points(series):
    discontinuous_points = []
    for i in range(1, len(series)):
        if series.iloc[i] - series.iloc[i-1] != pd.Timedelta(minutes=1):
            discontinuous_points.append(i)
    return discontinuous_points

# 시간이 끊긴 지점을 기준으로 데이터 나누기
break_points = find_discontinuous_points(pd.concat([x['date'] for x in X]))

# 60% 비율에 가장 근접한 끊긴 지점 찾기
val_ratio = 0.6
best_point = None
best_ratio = float('inf')
total_length = len(X)
for point in break_points:
    ratio = point / total_length
    if abs(ratio - val_ratio) < best_ratio:
        best_ratio = abs(ratio - val_ratio)
        val_best_point = point

# 80% 비율에 가장 근접한 끊긴 지점 찾기
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
    
    dec_train = dec[:val_best_point]
    dec_val = dec[val_best_point:test_best_point]
    dec_test = dec[test_best_point:]


X_train = [x.drop(columns=['date']) for x in X_train]
X_val = [x.drop(columns=['date']) for x in X_val]
X_test = [x.drop(columns=['date']) for x in X_test]

X_train_array = [x.values for x in X_train]
X_val_array = [x.values for x in X_val]
X_test_array = [x.values for x in X_test]

# MinMaxScaler 학습
scaler = MinMaxScaler().fit(np.concatenate(X_train_array, axis=0))

# X_train과 X_test 스케일링
X_train_scaled = [scaler.transform(x) for x in X_train_array]
X_val_scaled = [scaler.transform(x) for x in X_val_array]
X_test_scaled = [scaler.transform(x) for x in X_test_array] 

class Encoderlstm(Layer):
    def __init__(self, m):
        """
        m : feature dimension
        h0 : initial hidden state
        c0 : initial cell state
        """
        super(Encoderlstm, self).__init__(name="encoder_lstm")
        self.lstm = LSTM(m, return_state=True)
        self.initial_state = None

    def call(self, x, training=False):
        """
        x : t 번째 input data (shape = batch,1,n)
        """
        h_s, _, c_s = self.lstm(x, initial_state=self.initial_state)
        self.initial_state = [h_s, c_s]
        return h_s, c_s

    def reset_state(self, h0, c0):
        self.initial_state = [h0, c0]

#Encoder에서 사용되는 Input Attention은 예측하고자 하는 변수에 영향을 끼치는 외생변수들 중에 의미 있는 변수들에 attention 하여 사용하기 위해 적용
#각각 T의 시간 길이를 갖는 n개의 데이터를 사용하고 이를 Encoder에 있는 LSTM에 넣어줘서 hidden state를 뽑아줌
class InputAttention(Layer):
    def __init__(self, T):
        super(InputAttention, self).__init__(name="input_attention")
        self.w1 = Dense(T)
        self.w2 = Dense(T)
        self.v = Dense(1)

    def call(self, h_s, c_s, x):
        """
        h_s : hidden_state (shape = batch,m)
        c_s : cell_state (shape = batch,m)
        x : time series encoder inputs (shape = batch,T,n)
        """
        query = tf.concat([h_s, c_s], axis=-1)  # batch, m*2
        query = RepeatVector(x.shape[2])(query)  # batch, n, m*2
        x_perm = Permute((2, 1))(x)  # batch, n, T
        score = tf.nn.tanh(self.w1(x_perm) + self.w2(query))  # batch, n, T
        score = self.v(score)  # batch, n, 1
        score = Permute((2, 1))(score)  # batch,1,n
        attention_weights = tf.nn.softmax(score)  # t 번째 time step 일 때 각 feature 별 중요도
        return attention_weights

#모든 time step에 대해 attention weights를 구해준것을 받으면 input data와 곱해줘서 엑스햇티 구함
class Encoder(Layer):
    def __init__(self, T, m):
        super(Encoder, self).__init__(name="encoder")
        self.T = T
        self.input_att = InputAttention(T)
        self.lstm = Encoderlstm(m)
        self.initial_state = None
        self.alpha_t = None

    def call(self, data, h0, c0, n=39, training=False):
        """
        data : encoder data (shape = batch, T, n)
        n : data feature num
        """
        self.lstm.reset_state(h0=h0, c0=c0)
        alpha_seq = tf.TensorArray(tf.float32, self.T)
        for t in range(self.T):
            x = Lambda(lambda x: data[:, t, :])(data)
            x = x[:, tf.newaxis, :]  # (batch,1,n)

            h_s, c_s = self.lstm(x)

            self.alpha_t = self.input_att(h_s, c_s, data)  # batch,1,n

            alpha_seq = alpha_seq.write(t, self.alpha_t)
        alpha_seq = tf.reshape(alpha_seq.stack(), (-1, self.T, n))  # batch, T, n
        output = tf.multiply(data, alpha_seq)  # batch, T, n

        return output

#Attention 메커니즘이 적용된 변수 엑스햇티를 가지고 LSTM에 넣어준 후, 2번째 Attention인 Temporal attention을 적용
#Encoder에서 얻은 모든 Time step에서의 Hidden state와 각 time step에서의 Decoder LSTM의 hidden state를 비교하여 Attention 한 Context Vector를 추출
class TemporalAttention(Layer):
    def __init__(self, m):
        super(TemporalAttention, self).__init__(name="temporal_attention")
        self.w1 = Dense(m)
        self.w2 = Dense(m)
        self.v = Dense(1)

    def call(self, h_s, c_s, enc_h):
        """
        h_s : hidden_state (shape = batch,p)
        c_s : cell_state (shape = batch,p)
        enc_h : time series encoder inputs (shape = batch,T,m)
        """
        query = tf.concat([h_s, c_s], axis=-1)  # batch, p*2
        query = RepeatVector(enc_h.shape[1])(query)
        score = tf.nn.tanh(self.w1(enc_h) + self.w2(query))  # batch, T, m
        score = self.v(score)  # batch, T, 1
        attention_weights = tf.nn.softmax(
            score, axis=1
        )  # encoder hidden state h(i) 의 중요성 (0<=i<=T)
        return attention_weights

#y_{t} 값을 예측할 때는 이전 time step에서의 실제 y값과 Context Vector를 Concatenate해준 후 Dense layer를 거친 아웃풋 
#그리고 다음 Lstm에는 예측값을 넣어주며 최종 결과인 y_{T} 를 리턴
class Decoderlstm(Layer):
    def __init__(self, p):
        """
        p : feature dimension
        h0 : initial hidden state
        c0 : initial cell state
        """
        super(Decoderlstm, self).__init__(name="decoder_lstm")
        self.lstm = LSTM(p, return_state=True)
        self.initial_state = None

    def call(self, x, training=False):
        """
        x : t 번째 input data (shape = batch,1,n)
        """
        h_s, _, c_s = self.lstm(x, initial_state=self.initial_state)
        self.initial_state = [h_s, c_s]
        return h_s, c_s

    def reset_state(self, h0, c0):
        self.initial_state = [h0, c0]
class Decoder(Layer):
    def __init__(self, T, p, m):
        super(Decoder, self).__init__(name="decoder")
        self.T = T
        self.temp_att = TemporalAttention(m)
        self.dense = Dense(1)
        self.lstm = Decoderlstm(p)
        self.enc_lstm_dim = m
        self.dec_lstm_dim = p
        self.context_v = None
        self.dec_h_s = None
        self.beta_t = None

    def call(self, data, enc_h, h0=None, c0=None, training=False):
        """
        data : decoder data (shape = batch, T-1, 1)
        enc_h : encoder hidden state (shape = batch, T, m)
        """
        h_s = None
        self.lstm.reset_state(h0=h0, c0=c0)
        self.context_v = tf.zeros((enc_h.shape[0], 1, self.enc_lstm_dim))  # batch,1,m
        self.dec_h_s = tf.zeros((enc_h.shape[0], self.dec_lstm_dim))  # batch, p
        for t in range(self.T - 1):  # 0~T-1
            x = Lambda(lambda x: data[:, t, :])(data)
            x = x[:, tf.newaxis, :]  #  (batch,1,1)
            x = tf.concat([x, self.context_v], axis=-1)  # batch, 1, m+1
            x = self.dense(x)  # batch,1,1

            h_s, c_s = self.lstm(x)  # batch,p

            self.beta_t = self.temp_att(h_s, c_s, enc_h)  # batch, T, 1

            self.context_v = tf.matmul(
                self.beta_t, enc_h, transpose_a=True
            )  # batch,1,m
        return tf.concat(
            [h_s[:, tf.newaxis, :], self.context_v], axis=-1
        )  # batch,1,m+p

class DARNN(Model):
    def __init__(self, T, m, p):
        super(DARNN, self).__init__(name="DARNN")
        """
        T : 주기 (time series length)
        m : encoder lstm feature length
        p : decoder lstm feature length
        h0 : lstm initial hidden state
        c0 : lstm initial cell state
        """
        self.m = m
        self.encoder = Encoder(T=T, m=m)
        self.decoder = Decoder(T=T, p=p, m=m)
        self.lstm = LSTM(m, return_sequences=True)
        self.dense1 = Dense(p)
        self.dense2 = Dense(1)

    def call(self, inputs, training=False, mask=None):
        """
        inputs : [enc , dec]
        enc_data : batch,T,n
        dec_data : batch,T-1,1
        """
        enc_data, dec_data = inputs
        batch = enc_data.shape[0]
        h0 = tf.zeros((batch, self.m))
        c0 = tf.zeros((batch, self.m))
        enc_output = self.encoder(
            enc_data, n=39, h0=h0, c0=c0, training=training
        )  # batch, T, n
        enc_h = self.lstm(enc_output)  # batch, T, m
        dec_output = self.decoder(
            dec_data, enc_h, h0=h0, c0=c0, training=training
        )  # batch,1,m+p
        output = self.dense2(self.dense1(dec_output))
        output = tf.squeeze(output)
        return output

T = window_size
m = 16
p = 16
batch_size = 128

enc_dataset_train = np.array(X_train_scaled)
enc_dataset_val = np.array(X_val_scaled)
enc_dataset_test = np.array(X_test_scaled)

dec_dataset_train = np.array(dec_train)
dec_dataset_val = np.array(dec_val)
dec_dataset_test = np.array(dec_test)

target_train = np.array(y_train)
target_val = np.array(y_val)
target_test = np.array(y_test)

print(np.expand_dims(dec_dataset_train,axis=-1).shape)

class Dataloader(Sequence):
    
    def __init__(self, enc_dataset, dec_dataset, target_dataset, batch_size, shuffle=False):
        self.enc, self.dec, self.target = enc_dataset, dec_dataset, target_dataset
        self.batch_size = batch_size
        self.shuffle=shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.enc) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        
        batch_enc = np.array([self.enc[i] for i in indices])
        batch_dec = np.array([self.dec[i] for i in indices])
        batch_target = np.array([self.target[i] for i in indices])
        
        return [batch_enc, np.expand_dims(batch_dec, axis=-1)], batch_target

    # epoch이 끝날때마다 실행
    def on_epoch_end(self):
        self.indices = np.arange(len(self.enc))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
            
train_loader = Dataloader(enc_dataset_train, dec_dataset_train, target_train, batch_size, shuffle=True)
valid_loader = Dataloader(enc_dataset_val, dec_dataset_val, target_val, batch_size)
test_loader = Dataloader(enc_dataset_test, dec_dataset_test, target_test, batch_size)

model = DARNN(T=T, m=m, p=p)

train_ds = (
    tf.data.Dataset.from_tensor_slices(
        (enc_dataset_train, np.expand_dims(dec_dataset_train, axis=-1), target_train)
    )
    .batch(batch_size)
    .shuffle(buffer_size=len(enc_dataset_train))
    .prefetch(tf.data.experimental.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices(
        (enc_dataset_val, np.expand_dims(dec_dataset_val, axis=-1), target_val)
    )
    .batch(batch_size)
    .shuffle(buffer_size=len(enc_dataset_val))
    .prefetch(tf.data.experimental.AUTOTUNE)
)

@tf.function
def train_step(model, inputs, labels, loss_fn, optimizer, train_loss):
    with tf.GradientTape() as tape:
        prediction = model(inputs, training=True)
        loss = loss_fn(labels, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)


@tf.function
def test_step(model, inputs, labels, loss_fn, test_loss):
    with tf.GradientTape() as tape:
        prediction = model(inputs, training=False)
        loss = loss_fn(labels, prediction)
    test_loss(loss)
    return prediction


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print("Folder already exists, no action taken.")

folder_path = './DARNN/'
create_folder_if_not_exists(folder_path)

loss_fn = tf.keras.losses.MAE

optimizer = tf.keras.optimizers.Adam(0.001)
train_loss = tf.keras.metrics.Mean(name="train_loss")
val_loss = tf.keras.metrics.Mean(name="val_loss")

train_loss_list = []
val_loss_list = []

Epochs = 200

best_val_loss = np.inf
best_epoch = 0

for epoch in range(Epochs):
    for enc_data, dec_data, labels in train_ds:
        inputs = [enc_data, dec_data]
        train_step(model, inputs, labels, loss_fn, optimizer, train_loss)

    for enc_data, dec_data, labels in val_ds:
        inputs = [enc_data, dec_data]
        test_step(model, inputs, labels, loss_fn, val_loss)
    
    print(f"Epoch : {epoch + 1}, Train Loss : {train_loss.result()}, Val Loss : {val_loss.result()}")
    train_loss_list.append(train_loss.result())
    val_loss_list.append(val_loss.result())
    
    current_val_loss = val_loss.result()
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        best_epoch = epoch
        model.save_weights(folder_path + 'best_model.ckpt')  # Save model weights
        print(f"Best model saved at epoch {epoch+1}")
    
    train_loss.reset_states()
    val_loss.reset_states()

# Loss 시각화
plt.figure(figsize=(12, 6))
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Validation Loss')
plt.title("Train and Validation Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(folder_path + 'loss.png', dpi=300)
plt.clf()

test_ds = tf.data.Dataset.from_tensor_slices(
    (enc_dataset_test, np.expand_dims(dec_dataset_test, axis=-1), target_test)
).batch(len(target_test))
i = 0
test_loss = tf.keras.metrics.Mean(name="test_loss")

model.load_weights(folder_path + 'best_model.ckpt')

for enc_data, dec_data, label in test_ds:
    inputs = [enc_data, dec_data]
    pred = test_step(model, inputs, label, loss_fn, test_loss)
    if i == 0:
        preds = pred.numpy()
        labels = label.numpy()
        i += 1
    else:
        preds = np.concatenate([preds, pred.numpy()], axis=0)
        labels = np.concatenate([labels, label.numpy()], axis=0)
print("Test Loss: " + str(test_loss.result()))

all_predictions = np.array(preds)
all_labels = np.array(labels)

# Calculate metrics
mse = mean_squared_error(all_labels, all_predictions)
mae = mean_absolute_error(all_labels, all_predictions)
r2 = r2_score(all_labels, all_predictions)

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")

# 성능 지표를 텍스트 파일로 저장
with open(folder_path +'model_performance.txt', 'w') as file:
    file.write(f'MSE: {mse}\n')
    file.write(f'R2 Score: {r2}\n')

plt.figure(figsize=(12, 6))
plt.plot(all_labels, label="Actual Values", color='blue')
plt.plot(all_predictions, label="Predictions", color='red', linestyle='--')
plt.title("Time Series Prediction - DARNN")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.savefig(folder_path + 'Time Series Prediction.png', dpi=300)
plt.clf()

sns.jointplot(x=all_labels, y=all_predictions.flatten(), kind='reg')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.savefig(folder_path + 'Scatter Plot.png', dpi=300)
plt.clf()

# 타임스탭 관련 어텐션 스코어 플롯
enc_data, dec_data, label = next(iter(test_ds))
inputs = [enc_data, dec_data]

pred = model(inputs)
beta = []
for i in range(window_size):
    beta.append(np.mean(model.decoder.beta_t[:, i, 0].numpy()))
plt.bar(x=range(window_size), height=beta, color="orange")
plt.style.use("seaborn-pastel")
plt.title("Importance by Time Stamp")
plt.xlabel("time")
plt.ylabel("prob")
plt.savefig(folder_path + 'Time Stamp Attention Map.png', dpi=300)
plt.clf()

# 피쳐중요도 관련 어텐션 스코어 플롯
# 특성 이름
feature_names = X_train[0].columns.tolist()

pred = model(inputs)
alpha = []
for i in range(len(feature_names)):
    alpha.append(np.mean(model.encoder.alpha_t[:, 0, i].numpy()))

# 특성 중요도와 이름을 DataFrame으로 변환
feature_importances = pd.DataFrame({
    'feature': feature_names,
    'attention_score': alpha
})

# 중요도에 따라 특성을 정렬
feature_importances = feature_importances.sort_values(by='attention_score', ascending=False)

# 중요도에 따라 특성을 정렬하고 상위 10개를 선택
top_feature_importances = feature_importances.sort_values(by='attention_score', ascending=False).head(10)

# 바 차트로 상위 10개 특성 중요도를 시각화
top_feature_importances.plot(kind='bar', x='feature', y='attention_score', legend=False)
plt.title('Top 10 Feature Importances using Attention Score')
plt.ylabel('Average Impact on Model Output')
plt.xticks(rotation=45)  # 특성 이름이 길 경우 회전시켜서 라벨이 겹치지 않도록 설정
plt.savefig(folder_path + 'Feature Attention Map.png', dpi=300)