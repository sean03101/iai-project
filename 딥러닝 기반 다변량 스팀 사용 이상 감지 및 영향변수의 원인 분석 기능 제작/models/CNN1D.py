import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from collections import Counter

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix


class CNN1D(nn.Module):
    def __init__(self, input_channels, output_size):
        super(CNN1D, self).__init__()
        
        self.layers = nn.Sequential(
            # input: [batch_size, input_channels, 30(window size)] where L is the sequence length
            nn.Conv1d(input_channels, 16, kernel_size=3),  # output: [batch_size, 16, 28]
            nn.Tanh(),
            nn.Conv1d(16, 32, kernel_size=3),  # output: [batch_size, 32, 26]
            nn.Tanh(),
            nn.Conv1d(32, 64, kernel_size=3),  # output: [batch_size, 64, 24]
            nn.Tanh(),
            nn.Conv1d(64, 128, kernel_size=3), # output: [batch_size, 128, 22]
            nn.Tanh(),
            nn.Conv1d(128, 256, kernel_size=3), # output: [batch_size, 256, 20]
            nn.Tanh(),
            nn.Flatten(),  # output: [batch_size, 256 * 20]
            nn.Linear(256 * 20, 256),  # Adjust L based on actual input length
            nn.Tanh(),
            nn.Dropout(0.4),
            nn.Linear(256, 32),
            nn.Tanh(),
            nn.Dropout(0.4),
            nn.Linear(32, output_size)  # output_size is the final output size
        )
        
    def forward(self, x):
        return self.layers(x)
    

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print("Folder already exists, no action taken.")

def train_regression_1dcnn(model, train_dataloader, num_epochs, device, model_save_path):
    create_folder_if_not_exists('./1D_CNN_model/')
    criterion = nn.L1Loss()  # �ս� �Լ� ����
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # ��Ƽ������ ����
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_dataloader:

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
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
    plt.title("Train Loss 1D_CNN regression")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def test_regression_1dcnn(model, test_dataloader, device, y_test_date):
    model.load_state_dict(torch.load('./1D_CNN_model/last_model.pth'))
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)            
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

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
    date_list = np.concatenate([np.array([series['date']]) for series in y_test_date if 'date' in series])

    
    all_labels_class =  np.where(all_labels.flatten() > UCL_list, 2, np.where(all_labels.flatten() < LCL_list, 1, 0))
    all_predictions_class =  np.where(all_predictions.flatten() > UCL_list, 2, np.where(all_predictions.flatten() < LCL_list, 1, 0))

    unique, counts = np.unique(all_labels_class, return_counts=True)
    value_counts = dict(zip(unique, counts))

    print(value_counts)
    print(classification_report(all_predictions_class, all_labels_class, digits=4))

    return all_labels, all_predictions, UCL_list, LCL_list, date_list, all_labels_class, all_predictions_class


def visualiztion_1dcnn_regression_results(all_labels, all_predictions, UCL_list, LCL_list):
    """
    This function visualizes the results of the regression model. 
    - Confusion matrix 
    - Time Series Prediction Graphs 
    - A joint plot that shows the relationship between the actual value and the predicted value
    """
    
    def plot_confusion_matrix(con_mat, labels, title='Confusion Matrix_1DCNN', cmap=plt.cm.get_cmap('Blues'), normalize=False):
        """
        confusion matrix visualization
        """
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
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('1D_CNN_model/Confusion.png', dpi=300, bbox_inches='tight')
        plt.show()

    all_labels_class = np.where(all_labels.flatten() > UCL_list, 2, np.where(all_labels.flatten() < LCL_list, 1, 0))
    all_predictions_class = np.where(all_predictions.flatten() > UCL_list, 2, np.where(all_predictions.flatten() < LCL_list, 1, 0))
    cm = confusion_matrix(all_labels_class, all_predictions_class)
    labels = ['ok', 'Under LCL', 'Over UCL']
    plot_confusion_matrix(cm, labels=labels, normalize=False)

    plt.figure(figsize=(45, 10))
    plt.plot(all_labels, label="Actual Values", color='blue')
    plt.plot(all_predictions, label="Predictions", color='red', linestyle='--')
    plt.plot(UCL_list, color='black', linestyle='--', label='UCL(Based on yesterday)', linewidth=2)
    plt.plot(LCL_list, color='black', linestyle='--', label='LCL(Based on yesterday)', linewidth=2)
    plt.fill_between(range(len(all_labels)), LCL_list, UCL_list, color='grey', alpha=0.2)
    plt.title("Time Series Prediction_1D CNN")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig('1D_CNN_model/Time Series Prediction.png', dpi=300,  bbox_inches='tight')
    plt.show()

    sns.jointplot(x=all_labels.flatten(), y=all_predictions.flatten(), kind='reg')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.savefig('1D_CNN_model/Time Series Prediction jointplot.png', dpi=300, bbox_inches='tight')
    plt.show()


def train_classification_1dcnn(model, train_dataloader_clf, y_train, num_epochs, device, model_save_path):
    create_folder_if_not_exists('./1D_CNN_model_clf/')

    class_counts = Counter(y_train)
    total_samples = len(y_train)

    weights = {class_id: total_samples / (len(class_counts) * count) for class_id, count in class_counts.items()}
    weights_array = np.array([weights[class_id] for class_id in sorted(weights)])

    weights_normalized = weights_array / weights_array.sum()
    weights_normalized = torch.tensor(weights_normalized, dtype=torch.float)

    # 손실 함수에 클래스 가중치 적용
    loss_func = nn.CrossEntropyLoss(weight=weights_normalized.to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_loss_arr = []

    for epoch in range(num_epochs):
        model.train()
        train_running_loss = 0.0
        train_running_accuracy = 0.0
        total = 0 

        for X_batch, y_batch in train_dataloader_clf:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
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
        train_loss_arr.append(epoch_loss)  # 에포크별 평균 손실 저장

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.6f}, Train Accuracy: {epoch_accuracy:.2f}%")
        
    torch.save(model.state_dict(), model_save_path)

    # Loss 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss_arr, label='Train Loss')
    plt.title("Train Loss 1D_CNN classification")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def test_classification_1dcnn(model, test_dataloader, device):
    model.load_state_dict(torch.load('./1D_CNN_model_clf/last_model.pth'))
    model.eval()
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1) # 6개의 class중 가장 값이 높은 것을 예측 label로 추출.
            total += y_batch.size(0)                  # 예측값과 실제값이 맞으면 1 아니면 0으로 합산.
            correct += (predicted == y_batch).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    print(classification_report(all_predictions, all_labels, digits=4))
    return all_labels, all_predictions

def visualization_1dcnn_classification_results(all_labels, all_predictions, y_test_date):

    def plot_confusion_matrix(con_mat, labels, title='Confusion Matrix_1D CNN_classification', cmap=plt.cm.Blues, normalize=False):
        plt.clf()
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
        plt.savefig('1D_CNN_model_clf/Confusion.png',dpi=300, bbox_inches='tight')
        plt.show()

    # confusion matrix 시각화
    cm = confusion_matrix(all_labels, all_predictions)
    labels = ['ok', 'Under LCL', 'Over UCL']
    plot_confusion_matrix(cm, labels=labels, normalize=False)

    UCL_list = np.concatenate([np.array([series['UCL']]) for series in y_test_date if 'UCL' in series])
    LCL_list = np.concatenate([np.array([series['LCL']]) for series in y_test_date if 'LCL' in series])
    ei_list = np.concatenate([np.array([series['ei']]) for series in y_test_date if 'date' in series])

    # 0이 아닌 예측값과 해당 인덱스를 필터링
    normal_predictions = np.array(all_predictions) != 0
    indices = np.arange(len(all_predictions))[normal_predictions]
    normal_predictions = np.array(all_predictions)[normal_predictions]

    # 1과 2의 예측값과 해당 인덱스를 필터링
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
    plt.savefig('1D_CNN_model_clf/Time Series Prediction.png',dpi=300,  bbox_inches='tight')
    plt.show()
    plt.clf() 

