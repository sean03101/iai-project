import os
import itertools
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix


class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.attention_weight = nn.Parameter(torch.FloatTensor(feature_dim, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_in):
        attention_probs = self.softmax(torch.matmul(x_in, self.attention_weight).squeeze(2))
        weighted_sequence = torch.bmm(x_in.permute(0, 2, 1), attention_probs.unsqueeze(2)).squeeze(2)
        return weighted_sequence, attention_probs   # �ð�ȭ�� ���� ���ټ� ���ھ�� weight ���ÿ� ����


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
    
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print("Folder already exists, no action taken.")

def train_regression_LSTM(model, train_dataloader, num_epochs, device, model_save_path):
    create_folder_if_not_exists('./LSTM/')
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train_losses = []
    for epoch in range(num_epochs):
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
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}")

    torch.save(model.state_dict(), model_save_path)
    # Loss �ð�ȭ
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.title("Train Loss LSTM regression")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def test_regression_LSTM(model, test_dataloader, device, y_test_date):
    model.load_state_dict(torch.load('./LSTM/last_model.pth'))
    model.eval()
    all_predictions = []
    all_labels = []
    all_attention_maps = []

    with torch.no_grad():
        for X_batch, y_batch in test_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs, attention_map = model(X_batch)
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            all_attention_maps.append(attention_map.cpu().numpy())

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

    return all_attention_maps, all_labels, all_predictions, UCL_list, LCL_list, all_labels_class, all_predictions_class

def visualiztion_LSTM_regression_results(all_attention_maps, all_predictions, all_labels, UCL_list, LCL_list, all_predictions_class, all_labels_class):
    # ���ټ� �� �ð�ȭ
    avg_attention_weights = np.concatenate(all_attention_maps, axis=0).mean(axis=0)
    plt.figure(figsize=(15, 5))
    plt.plot(avg_attention_weights)
    plt.xlabel("Time Steps")
    plt.ylabel("Average Attention Weight")
    plt.title("Average Attention Map across all test instances")
    plt.savefig('LSTM/Attention all.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ȥ�� ��� �ð�ȭ
    def plot_confusion_matrix(con_mat, labels, title='Confusion Matrix_LSTM', cmap=plt.cm.get_cmap('Blues'), normalize=False):
        plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
        plt.title(title)
        marks = np.arange(len(labels))
        nlabels = ['{0}(n={1})'.format(label, sum(con_mat[i])) for i, label in enumerate(labels)]
        plt.xticks(marks, labels, rotation=45)
        plt.yticks(marks, nlabels)

        thresh = con_mat.max() / 2.
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            if normalize:
                plt.text(j, i, '{0}%'.format(int(con_mat[i, j] * 100 / sum(con_mat[i]))), horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
            else:
                plt.text(j, i, con_mat[i, j], horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('Predicted label')
        plt.xlabel('True label')
        plt.savefig('LSTM/Confusion.png', dpi=300, bbox_inches='tight')
        plt.show()

    cm_df = confusion_matrix(all_labels_class, all_predictions_class)
    labels = ['ok', 'Under LCL', 'Over UCL']
    plot_confusion_matrix(cm_df, labels=labels, normalize=False)

    # Jointplot 시각화
    sns.jointplot(x=all_labels, y=all_predictions.flatten(), kind='reg')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.savefig('IMV_LSTM/Time Series Prediction jointplot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # �ð迭 �ð�ȭ
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
    plt.savefig('LSTM/Time Series Prediction.png', dpi=300,  bbox_inches='tight')
    plt.show()
    plt.clf()

    
def train_classification_LSTM(model, train_dataloader_clf, y_train, num_epochs, device, model_save_path):
    create_folder_if_not_exists('./LSTM_clf/')

    class_counts = Counter(y_train)
    total_samples = len(y_train)

    weights = {class_id: total_samples / (len(class_counts) * count) for class_id, count in class_counts.items()}
    weights_array = np.array([weights[class_id] for class_id in sorted(weights)])

    weights_normalized = weights_array / weights_array.sum()
    weights_normalized = torch.tensor(weights_normalized, dtype=torch.float)

    loss_func = nn.CrossEntropyLoss(weight=weights_normalized.to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    train_loss_arr = []
    n = len(train_dataloader_clf)
 
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_running_loss = 0.0
        train_running_accuracy = 0.0
        total = 0 

        for X_batch, y_batch in train_dataloader_clf:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs, __ = model(X_batch)
            loss = loss_func(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_running_loss += loss.item()

            total += y_batch.size(0)
            _, predicted = torch.max(outputs, 1) 
            train_running_accuracy += (predicted == y_batch).sum().item()

        train_loss_arr.append(train_running_loss / n)
        train_accuracy = (100 * train_running_accuracy / total)  

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_running_loss/len(train_dataloader_clf):.6f}, train_accuracy: {train_accuracy:.2f}")  

    # Save the model
    torch.save(model.state_dict(), model_save_path)
    # Loss �ð�ȭ
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss_arr, label='Train Loss')
    plt.title("Train Loss LSTM classisification")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()  


def test_classification_LSTM(model, test_dataloader_clf, device):
    model.load_state_dict(torch.load('./LSTM_clf/last_model.pth'))
    model.eval()
    all_predictions = []
    all_labels = []
    all_attention_maps = []

    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_dataloader_clf:
            
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs, attention_map = model(X_batch)
            
            all_labels.append(y_batch.cpu().numpy())
            
            _, predicted = torch.max(outputs.data, 1) 
            total += y_batch.size(0) # test ����
            correct += (predicted == y_batch).sum().item() # �������� �������� ������ 1 �ƴϸ� 0���� �ջ�.
            all_predictions.append(predicted.cpu().numpy())
            all_attention_maps.append(attention_map.cpu().numpy())
            
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    # Convert to numpy arrays
    all_labels = np.concatenate(all_labels)
    all_predictions_clf = np.concatenate(all_predictions) 

    # Print classification report
    print(classification_report(all_predictions_clf, all_labels, digits=4))
    return all_labels, all_predictions_clf, all_attention_maps

def visualiztion_LSTM_classification_results(all_attention_maps, all_predictions, all_labels, y_test_date):
    # ���ټ� �� �ð�ȭ
    print('attention map')
    all_attention_maps = np.concatenate(all_attention_maps, axis=0)
    avg_attention_weights = all_attention_maps.mean(axis=0)

    plt.figure(figsize=(15, 5))
    plt.plot(avg_attention_weights)
    plt.xlabel("Time Steps")
    plt.ylabel("Average Attention Weight")
    plt.title("Average Attention Map across all test instances")
    plt.savefig('LSTM_clf/Attention all.png',dpi=300, bbox_inches='tight')
    plt.show()

    # ȥ�� ��� �ð�ȭ
    print('confusion matrix')
    def plot_confusion_matrix(con_mat, labels, title='Confusion Matrix_LSTM', cmap=plt.cm.Blues, normalize=False):
        plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
        plt.title(title)
        marks = np.arange(len(labels))
        nlabels = []
        for k in range(len(con_mat)):
            n = sum(con_mat[k])
            nlabel = '{0}(n={1})'.format(labels[k],n)
            nlabels.append(nlabel)
        plt.xticks(marks, labels)
        plt.yticks(marks, nlabels)

        thresh = con_mat.max() / 2.
        if normalize:
            for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
                plt.text(j, i, '{0}%'.format(con_mat[i, j] * 100 / n), horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
        else:
            for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
                plt.text(j, i, con_mat[i, j], horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
        plt.ylabel('Predicted label')
        plt.xlabel('True label')
        plt.tight_layout()

    cm = confusion_matrix(all_predictions, all_labels)
    plot_confusion_matrix(cm, labels=['ok', 'Under LCL', 'Over UCL'], normalize=False)
    plt.savefig(f'LSTM_clf/Confusion.png', dpi=300)
    plt.show()
    plt.clf()
    
    # �ð迭 �ð�ȭ
    print('Time Series plot')
    UCL_list = np.concatenate([np.array([series['UCL']]) for series in y_test_date if 'UCL' in series])
    LCL_list = np.concatenate([np.array([series['LCL']]) for series in y_test_date if 'LCL' in series])
    ei_list = np.concatenate([np.array([series['ei']]) for series in y_test_date if 'date' in series])

    # 0�� �ƴ� �������� �ش� �ε����� ���͸�
    normal_predictions = np.array(all_predictions) != 0
    indices = np.arange(len(all_predictions))[normal_predictions]
    normal_predictions = np.array(all_predictions)[normal_predictions]

    # 1�� 2�� �������� �ش� �ε����� ���͸�
    predictions_array = np.array(all_predictions)
    under_LCL_indices = np.where(predictions_array == 1)[0]
    under_LCL = predictions_array[under_LCL_indices]

    over_UCL_indices = np.where(predictions_array == 2)[0]
    over_UCL = predictions_array[over_UCL_indices]

    plt.figure(figsize=(45, 10))
    plt.plot(ei_list, label="Actual Values", color='blue')
    plt.scatter(under_LCL_indices, under_LCL, label="Predictions (under LCL)", color='red')
    plt.scatter(over_UCL_indices, over_UCL, label="Predictions (over UCL)", color='orange')
    plt.plot(UCL_list, color='black', linestyle='--', label=f'UCL(Based on yesterday)', linewidth=2)
    plt.plot(LCL_list, color='black', linestyle='--', label=f'LCL(Based on yesterday)', linewidth=2)
    plt.fill_between(range(len(all_labels)), LCL_list, UCL_list, color='grey', alpha=0.2)

    plt.title("Time Series Actual values")
    plt.ylabel("Value")
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('LSTM_clf/Time Series Prediction.png',dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf() 