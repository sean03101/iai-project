import os
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix

import sys

sys.path.append('models')

from shap_visualization import shap_image_plot

## IMV LSTM model --------------------------------------------------------------------------------------------

class IMVTensorLSTM(nn.Module):
    
    def __init__(self, input_dim, output_dim, n_units, init_std=0.02):
        super().__init__()
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.U_i = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.U_f = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.U_o = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.W_i = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.W_f = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.W_o = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_i = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_f = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_o = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1)*init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1)*init_std)
        self.F_beta = nn.Linear(2*n_units, 1)
        self.Phi = nn.Linear(2*n_units, output_dim)
        self.n_units = n_units
        self.input_dim = input_dim
    
    def forward(self, x):
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(x.device)
        c_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(x.device)
        outputs = []
        for t in range(x.shape[1]):
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_j) + self.b_j)
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) + \
                                      torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_i) + self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) + \
                                      torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_f) + self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) + \
                                      torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_o) + self.b_o)
            c_tilda_t = c_tilda_t*f_tilda_t + i_tilda_t*j_tilda_t
            h_tilda_t = (o_tilda_t*torch.tanh(c_tilda_t))
            outputs.append(h_tilda_t)
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) + self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas/torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas*outputs, dim=1)
        hg = torch.cat([g_n, h_tilda_t], dim=2)
        mu = self.Phi(hg)
        betas = torch.tanh(self.F_beta(hg))
        betas = torch.exp(betas)
        betas = betas/torch.sum(betas, dim=1, keepdim=True)
        mean = torch.sum(betas*mu, dim=1)
        
        return mean, alphas, betas


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print("Folder already exists, no action taken.")

## train imvLSTM regression --------------------------------------------------------------------------------------------

def train_regression_imvLSTM(model, train_dataloader, num_epochs, device, model_save_path):
    create_folder_if_not_exists('./IMV_LSTM/')
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs, alphas, betas = model(X_batch)
            
            loss = criterion(outputs.squeeze(), y_batch.squeeze())
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_dataloader.dataset)
        train_losses.append(train_loss)
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}")

    torch.save(model.state_dict(), './IMV_LSTM/last_model.pth')
    # Loss �ð�ȭ
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.title("Train Loss IMV LSTM regression")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

## test imvLSTM regression --------------------------------------------------------------------------------------------

def test_regression_imvLSTM(model, test_dataloader,device, y_test_date):
    model.load_state_dict(torch.load('./IMV_LSTM/last_model.pth'))
    model.eval()
    all_predictions = []
    all_labels = []
    all_alphas = []
    all_betas = []

    with torch.no_grad():
        for X_batch, y_batch in test_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs, alpha, beta = model(X_batch)
            
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            all_alphas.append(alpha.cpu().numpy())
            all_betas.append(beta.cpu().numpy())

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

    UCL_list = np.concatenate([np.array([series['UCL']]) for series in y_test_date if 'UCL' in series])
    LCL_list = np.concatenate([np.array([series['LCL']]) for series in y_test_date if 'LCL' in series])
    all_labels_class = np.where(all_labels.flatten() > UCL_list, 2, np.where(all_labels.flatten() < LCL_list, 1, 0))
    all_predictions_class = np.where(all_predictions.flatten() > UCL_list, 2, np.where(all_predictions.flatten() < LCL_list, 1, 0))

    unique, counts = np.unique(all_labels_class, return_counts=True)
    value_counts = dict(zip(unique, counts))

    print(value_counts)
    print(classification_report(all_predictions_class, all_labels_class, digits=4))
    return all_labels, all_predictions, UCL_list, LCL_list, all_labels_class, all_predictions_class, all_alphas, all_betas


## visualization imvLSTM regression --------------------------------------------------------------------------------------------

def visualiztion_imvLSTM_regression_results(all_labels, all_predictions, UCL_list, LCL_list, all_labels_class, all_predictions_class, all_alphas, all_betas, X_train_scaled, X_train):

    def plot_confusion_matrix(con_mat, labels, title='Confusion Matrix_IMV-LSTM', cmap=plt.cm.get_cmap('Blues'), normalize=False):
        plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
        plt.title(title)
        marks = np.arange(len(labels))
        nlabels = [f'{label}(n={sum(con_mat[i])})' for i, label in enumerate(labels)]
        plt.xticks(marks, labels)
        plt.yticks(marks, nlabels)

        thresh = con_mat.max() / 2.
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            if normalize:
                plt.text(j, i, f'{con_mat[i, j] * 100 / sum(con_mat[i]):.0f}%', horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
            else:
                plt.text(j, i, str(con_mat[i, j]), horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
        plt.ylabel('Predicted label')
        plt.xlabel('True label')
        plt.tight_layout()
        plt.savefig('IMV_LSTM/Confusion.png', dpi=300,  bbox_inches='tight')
        plt.show()

    cm_df = confusion_matrix(all_predictions_class, all_labels_class)    
    labels = ['ok', 'Under LCL', 'Over UCL']
    plot_confusion_matrix(cm_df, labels=labels, normalize=False)

    sns.jointplot(x=all_labels, y=all_predictions.flatten(), kind='reg')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.savefig('IMV_LSTM/Time Series Prediction jointplot.png', dpi=300,  bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(45, 10))
    plt.plot(all_labels, label="Actual Values", color='blue')
    plt.plot(all_predictions, label="Predictions", color='red', linestyle='--')
    plt.plot(UCL_list, color='black', linestyle='--', label='UCL (Based on yesterday)', linewidth=2)
    plt.plot(LCL_list, color='black', linestyle='--', label='LCL (Based on yesterday)', linewidth=2)
    plt.fill_between(range(len(all_labels)), LCL_list, UCL_list, color='grey', alpha=0.2)
    plt.title("Time Series Actual values")
    plt.ylabel("Value")
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('IMV_LSTM/Time Series Prediction.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()

    alphas = np.concatenate(all_alphas).mean(axis=0)[..., 0].transpose(1, 0)
    betas = np.concatenate(all_betas).mean(axis=0)[..., 0]

    
    fig, ax = plt.subplots(figsize=(24, 20))
    im = ax.imshow(alphas, cmap=plt.cm.viridis)

    # 틱 라벨 설정
    ax.set_xticks(np.arange(np.array(X_train_scaled).shape[1]))
    ax.set_yticks(np.arange(len(X_train[0].columns)))
    ax.set_xticklabels(["t-"+str(30-i) for i in np.arange(np.array(X_train_scaled).shape[1]-1, -1, -1)], fontsize=15)
    ax.set_yticklabels(list(X_train[0].columns), fontsize=15)

    # 셀 값 표시
    for i in range(len(X_train[0].columns)):
        for j in range(np.array(X_train_scaled).shape[1]):
            text = ax.text(j, i, round(alphas[i, j], 3),
                        ha="center", va="center", color="w")

    # 제목 설정
    ax.set_title("Temporal level attention map", fontsize=25)

    # 색상 막대 추가
    fig.colorbar(im, ax=ax)

    # 여백 조정
    fig.tight_layout()
    
    plt.savefig('IMV_LSTM/IMV_temporal_level_attentionmap.png', dpi=300,  bbox_inches='tight')
    plt.show()
    plt.clf()


    feature_names = X_train[0].columns.tolist()

    # 특성 중요도와 이름을 DataFrame으로 변환합니다.
    feature_importances = pd.DataFrame({
        'feature': feature_names,
        'importance': betas
    })

    # 중요도에 따라 특성을 정렬합니다.
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)
    plt.figure(figsize=(20, 20))
    # 중요도에 따라 특성을 정렬하고 상위 10개를 선택합니다.
    top_feature_importances = feature_importances.sort_values(by='importance', ascending=False).head(10)

    # 바 차트로 상위 10개 특성 중요도를 시각화합니다.
    top_feature_importances.plot(kind='bar', x='feature', y='importance', legend=False)
    plt.title('Top 10 Feature Importances using variable level attention')
    plt.ylabel('Average Impact on Model Output')
    plt.xticks(rotation=45)  # 특성 이름이 길 경우 회전시켜서 라벨이 겹치지 않도록 합니다.
    fig.tight_layout()  
    plt.savefig('IMV_LSTM/IMV_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

## train imvLSTM classification --------------------------------------------------------------------------------------------

def train_classification_imvLSTM(model, train_dataloader_clf, y_train, num_epochs, device, model_save_path):
    class_counts = Counter(y_train)
    total_samples = len(y_train)

    weights = {class_id: total_samples / (len(class_counts) * count) for class_id, count in class_counts.items()}
    weights_array = np.array([weights[class_id] for class_id in sorted(weights)])

    weights_normalized = weights_array / weights_array.sum()
    weights_normalized = torch.tensor(weights_normalized, dtype=torch.float)

    loss_func = nn.CrossEntropyLoss(weight=weights_normalized.to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train_loss_arr = []
    n = len(train_dataloader_clf)
    create_folder_if_not_exists('./IMV_LSTM_clf/')

    for epoch in range(num_epochs):
        model.train()
        train_running_loss = 0.0
        train_running_accuracy = 0.0
        total = 0

        for X_batch, y_batch in train_dataloader_clf:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device, dtype=torch.long)

            outputs, alphas, betas = model(X_batch)
            loss = loss_func(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_running_loss += loss.item()
            total += y_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            train_running_accuracy += (predicted == y_batch).sum().item()

        epoch_loss = train_running_loss / len(train_dataloader_clf)
        epoch_accuracy = (100 * train_running_accuracy / total)
        train_loss_arr.append(epoch_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.6f}, Train Accuracy: {epoch_accuracy:.2f}%")
    torch.save(model.state_dict(), './IMV_LSTM_clf/last_model.pth')

    plt.figure(figsize=(12, 6))
    plt.plot(train_loss_arr, label='Train Loss')
    plt.title("Train Loss IMV LSTM classisification")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

## test imvLSTM classification --------------------------------------------------------------------------------------------

def test_classification_imvLSTM(model, test_dataloader_clf, device):
    model.load_state_dict(torch.load('./IMV_LSTM_clf/last_model.pth'))
    model.eval()
    all_predictions = []
    all_labels = []
    all_alphas = []
    all_betas = []
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_dataloader_clf:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device, dtype=torch.long)

            outputs, alpha, beta = model(X_batch)
            
            _, predicted = torch.max(outputs.data, 1)  
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()  
            
            all_labels.append(y_batch.cpu().numpy())
            all_alphas.append(alpha.cpu().numpy())
            all_betas.append(beta.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
            
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    all_predictions_clf = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)

    # ���� ��
    print(classification_report(all_predictions_clf, all_labels, digits=4))
    return all_labels, all_predictions_clf, all_alphas, all_betas

## visualization imvLSTM classification --------------------------------------------------------------------------------------------

def visualiztion_imvLSTM_classification_results(all_predictions, all_labels, all_alphas, all_betas,X_train_scaled, X_train, y_test_date):

    def plot_confusion_matrix(con_mat, labels, title='Confusion Matrix_IMV-LSTM', cmap=plt.cm.get_cmap('Blues'), normalize=False):
        plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
        plt.title(title)
        marks = np.arange(len(labels))
        nlabels = [f'{label}(n={sum(con_mat[i])})' for i, label in enumerate(labels)]
        plt.xticks(marks, labels)
        plt.yticks(marks, nlabels)

        thresh = con_mat.max() / 2.
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            if normalize:
                plt.text(j, i, f'{con_mat[i, j] * 100 / sum(con_mat[i]):.0f}%', horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
            else:
                plt.text(j, i, str(con_mat[i, j]), horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
        plt.ylabel('Predicted label')
        plt.xlabel('True label')
        plt.tight_layout()
        plt.savefig('IMV_LSTM_clf/Confusion.png', dpi=300,  bbox_inches='tight')
        plt.show()

    cm_df = confusion_matrix(all_predictions, all_labels)    
    labels = ['ok', 'Under LCL', 'Over UCL']
    plot_confusion_matrix(cm_df, labels=labels, normalize=False)

    print('Time Series plot')
    UCL_list, LCL_list, date_list, ei_list = [np.concatenate([np.array([series[key]]) for series in y_test_date if key in series]) for key in ['UCL', 'LCL', 'date', 'ei']]
    
    normal_predictions = np.array(all_predictions) != 0
    indices = np.arange(len(all_predictions))[normal_predictions]
    normal_predictions = np.array(all_predictions)[normal_predictions]

    predictions_array = np.array(all_predictions)
    under_LCL_indices = np.where(predictions_array == 1)[0]
    under_LCL = predictions_array[under_LCL_indices]

    over_UCL_indices = np.where(predictions_array == 2)[0]
    over_UCL = predictions_array[over_UCL_indices]

    plt.figure(figsize=(45, 10))
    plt.plot(ei_list, label="Actual Values", color='blue')
    plt.scatter(under_LCL_indices, under_LCL, label="Predictions (under LCL)", color='red')
    plt.scatter(over_UCL_indices, over_UCL, label="Predictions (over UCL)", color='orange')
    plt.plot(UCL_list, color='black', linestyle='--', label='UCL (Based on yesterday)', linewidth=2)
    plt.plot(LCL_list, color='black', linestyle='--', label='LCL (Based on yesterday)', linewidth=2)
    plt.fill_between(range(len(ei_list)), LCL_list, UCL_list, color='grey', alpha=0.2)
    plt.title("Time Series Actual values")
    plt.ylabel("Value")
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('IMV_LSTM_clf/Time Series Prediction.png', dpi=300,  bbox_inches='tight')
    plt.show()

    print('feature importance')
    alphas = np.concatenate(all_alphas).mean(axis=0)[..., 0].transpose(1, 0)
    betas = np.concatenate(all_betas).mean(axis=0)[..., 0]

    fig, ax = plt.subplots(figsize=(24, 20))
    im = ax.imshow(alphas, cmap=plt.cm.viridis)

    # 틱 라벨 설정
    ax.set_xticks(np.arange(np.array(X_train_scaled).shape[1]))
    ax.set_yticks(np.arange(len(X_train[0].columns)))
    ax.set_xticklabels(["t-"+str(30-i) for i in np.arange(np.array(X_train_scaled).shape[1]-1, -1, -1)], fontsize=15)
    ax.set_yticklabels(list(X_train[0].columns), fontsize=15)

    # 셀 값 표시
    for i in range(len(X_train[0].columns)):
        for j in range(np.array(X_train_scaled).shape[1]):
            text = ax.text(j, i, round(alphas[i, j], 3),
                        ha="center", va="center", color="w")

    # 제목 설정
    ax.set_title("Temporal level attention map", fontsize=25)

    # 색상 막대 추가
    fig.colorbar(im, ax=ax)

    # 여백 조정
    fig.tight_layout()
    plt.savefig('IMV_LSTM_clf/IMV_temporal_level_attentionmap.png', dpi=300,  bbox_inches='tight')
    plt.show()

    feature_names = X_train[0].columns.tolist()

    # 특성 중요도와 이름을 DataFrame으로 변환합니다.
    feature_importances = pd.DataFrame({
        'feature': feature_names,
        'importance': betas
    })

    # 중요도에 따라 특성을 정렬합니다.
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)

    # 중요도에 따라 특성을 정렬하고 상위 10개를 선택합니다.
    top_feature_importances = feature_importances.sort_values(by='importance', ascending=False).head(10)
    plt.figure(figsize=(20, 20))
    # 바 차트로 상위 10개 특성 중요도를 시각화합니다.
    top_feature_importances.plot(kind='bar', x='feature', y='importance', legend=False)
    plt.title('Top 10 Feature Importances using variable level attention')
    plt.ylabel('Average Impact on Model Output')
    plt.xticks(rotation=45)  # 특성 이름이 길 경우 회전시켜서 라벨이 겹치지 않도록 합니다.
    fig.tight_layout()
    plt.savefig('IMV_LSTM_clf/IMV_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()