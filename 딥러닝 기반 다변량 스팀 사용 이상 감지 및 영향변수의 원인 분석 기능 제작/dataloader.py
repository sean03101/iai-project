import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

## LSTM, IMV-LSTM dataloader_regression --------------------------------------------------------------------------------------------

def create_dataloaders_regression(X_train_scaled, y_train, X_test_scaled, y_test, batch_size_train=128, batch_size_test=64, shuffle_train=True, shuffle_test=False):
    '''
    Parameters
    ----------
    X_train_scaled : [numpy.array]
        train dataset's input

    y_train : list or np.ndarray
        train dataset's label(ei)

    X_test_scaled : [numpy.array]
        test dataset's input
        
    y_test: list or np.ndarray
        test dataset's label(ei)

    batch_size_train : int
        train dataloader's batch size

    batch_size_test : int
        test dataloader's batch size

    shuffle_train : bool
        Whether train data will be shuffled into the data loader
        
    shuffle_test : bool
        Whether test data will be shuffled into the data loader
        
    output : train & test regression dataloader / the shape of the input batch for the model is [batch_size, input dataset's column, time stamp]
    '''
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=shuffle_train)

    
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=shuffle_test)

    return train_dataloader, test_dataloader


def create_dataloaders_classificiation(X_train_scaled, y_train, X_test_scaled, y_test, batch_size_train=128, batch_size_test=64, shuffle_train=True, shuffle_test=False):

    '''
    Parameters
    ----------
    X_train_scaled : [numpy.array]
        train dataset's input

    y_train : list or np.ndarray
        train dataset's label(is_abnormal)

    X_test_scaled : [numpy.array]
        test dataset's input
        
    y_test: list or np.ndarray
        test dataset's label(is_abnormal)

    batch_size_train : int
        train dataloader's batch size

    batch_size_test : int
        test dataloader's batch size

    shuffle_train : bool
        Whether train data will be shuffled into the data loader
        
    shuffle_test : bool
        Whether test data will be shuffled into the data loader
        
    output : train & test regression dataloader / the shape of the input batch for the model is [batch_size, input dataset's column, time stamp]
    '''
    
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=shuffle_train)

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64) 
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=shuffle_test)

    return train_dataloader, test_dataloader

## 1D CNN dataloader --------------------------------------------------------------------------------------------

def create_cnn_dataloaders_regression(X_train_scaled, y_train, X_test_scaled, y_test, batch_size_train=128, batch_size_test=64, shuffle_train=True, shuffle_test=False):
    '''
    Parameters
    ----------
    X_train_scaled : [numpy.array]
        train dataset's input

    y_train : list or np.ndarray
        train dataset's label(ei)

    X_test_scaled : [numpy.array]
        test dataset's input
        
    y_test: list or np.ndarray
        test dataset's label(ei)

    batch_size_train : int
        train dataloader's batch size

    batch_size_test : int
        test dataloader's batch size

    shuffle_train : bool
        Whether train data will be shuffled into the data loader
        
    shuffle_test : bool
        Whether test data will be shuffled into the data loader
        
    output : train & test regression dataloader / the shape of the input batch for the model is [batch_size, time stamp ,input dataset's column]
    '''
    X_train_tensor = torch.tensor(np.array(X_train_scaled), dtype=torch.float32).permute(0, 2, 1)
    y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    X_test_tensor = torch.tensor(np.array(X_test_scaled), dtype=torch.float32).permute(0, 2, 1)
    y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.float32)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=shuffle_train)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=shuffle_test)

    return train_dataloader, test_dataloader, X_train_tensor, X_test_tensor

def create_cnn_dataloaders_classification(X_train_scaled, y_train, X_test_scaled, y_test, batch_size_train=128, batch_size_test=64, shuffle_train=True, shuffle_test=False):

    '''
    Parameters
    ----------
    X_train_scaled : [numpy.array]
        train dataset's input

    y_train : list or np.ndarray
        train dataset's label(is_abnormal)

    X_test_scaled : [numpy.array]
        test dataset's input
        
    y_test: list or np.ndarray
        test dataset's label(is_abnormal)

    batch_size_train : int
        train dataloader's batch size

    batch_size_test : int
        test dataloader's batch size

    shuffle_train : bool
        Whether train data will be shuffled into the data loader
        
    shuffle_test : bool
        Whether test data will be shuffled into the data loader
        
    output : train & test regression dataloader / the shape of the input batch for the model is [batch_size, time stamp ,input dataset's column]
    '''
    X_train_tensor = torch.tensor(np.array(X_train_scaled), dtype=torch.float32).permute(0, 2, 1)
    y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.int64)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    X_test_tensor = torch.tensor(np.array(X_test_scaled), dtype=torch.float32).permute(0, 2, 1)
    y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.int64)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=shuffle_train)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=shuffle_test)

    return train_dataloader, test_dataloader, X_train_tensor, X_test_tensor
