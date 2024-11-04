from PIL import Image
import torch
import torch.nn as nn
import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os
import cv2


def gray_data(data_X):
    gray_images_X = []
    for img in data_X['image'].values:
        if isinstance(img, Image.Image):
            gray_image = img.convert("L")
            gray_images_X.append(np.array(gray_image))
        else:
            if img.ndim == 3 and img.shape[-1] == 3:
                gray_image = np.mean(img, axis=-1)
                gray_images_X.append(gray_image)
            else:
                gray_images_X.append(img)

    return gray_images_X

def load_image(file_name, folder_path):

    image_path = os.path.join(folder_path, file_name)
    image = cv2.imread(image_path)
    return image

def add_image_column(df, folder_path, special_name):
    df['image'] = df['filename'].apply(lambda file_name: load_image(special_name+"_"+file_name, folder_path))
    return df

def count_similiarity(y_pred, y):

    '''
    It counts similarity between predicted value and true value.
    :param y_pred: predicted values
    :param y: real values
    :return: similarity value float
    '''

    count = 0
    for num1, num2 in zip(y_pred, y):

        if num1 == num2:
            count += 1

    return count


def split_tensor_data(data_X, data_val_X, batch_size=10):

    gray_images_X = gray_data(data_X)
    X_train = np.stack(gray_images_X)
    y_train = data_X['contains_object'].astype(int).values

    gray_images_X = gray_data(data_val_X)
    X_val = np.stack(gray_images_X)
    y_val = data_val_X['contains_object'].astype(int).values

    X_train = torch.tensor(X_train, dtype=torch.float)
    X_val = torch.tensor(X_val, dtype=torch.float)

    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)

    return train_loader, val_loader

def tensor_to_rounded_list(tensor, decimal_places=2):
    values_list = tensor.squeeze().tolist()
    rounded_list = [round(val, decimal_places) for val in values_list]
    return rounded_list


class Simple_CNN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8, output_dim=1):
        '''
        Function which creates a CNN with specified input dimensions and output dimensions.

        :param input_dim: number of input channels (int)
        :param output_dim: number of output classes (int)
        '''
        super().__init__()  # Poprawione użycie super()

        self.seq_nn = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1, stride=1),  # (1, 2048, 2048) -> (8, 2048, 2048)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (8, 2048, 2048) -> (8, 1024, 1024)

            nn.Conv2d(hidden_dim, 2 * hidden_dim, kernel_size=3, padding=1, stride=1),  # (8, 1024, 1024) -> (16, 1024, 1024)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (16, 1024, 1024) -> (16, 512, 512)

            nn.Conv2d(2 * hidden_dim, 4 * hidden_dim, kernel_size=3, padding=1, stride=1),  # (16, 512, 512) -> (32, 512, 512)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (32, 512, 512) -> (32, 256, 256)

            nn.Conv2d(4 * hidden_dim, 8 * hidden_dim, kernel_size=3, padding=1, stride=1),  # (32, 256, 256) -> (64, 256, 256)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (64, 256, 256) -> (64, 128, 128)

            nn.Flatten(),
            nn.Linear(128 * 128 * 8 * hidden_dim, 512),  # Dopasowanie do wyjścia warstwy splotowej
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, X):
        return self.seq_nn(X)



def train(model_nn, train_loader: torch, val_loader: torch, epochs=100, lr=0.01, optimizer=torch.optim.Adam,
          choosen_loss_function=nn.MSELoss, count_similiarity=count_similiarity, model_name='new_model'):

    '''

    :param model_nn: pytorch model
    :param train_loader: training data as torch DataLoader
    :param val_loader: validation data as torch DataLoader
    :param epochs: number of epochs (loops)
    :param lr: learning rate (speed of learning)
    :param optimizer: type of optimizer (SGD, Adam)
    :param choosen_loss_function: type of loss function (nn.CrossEntropyLoss)
    :param count_similiarity: prepared function to count good and bad similiarity
    :return: train_metrics, val_metrics, best_model
    '''

    # creating metrics
    # it returns two dictionaries
    # each for train and val data

    train_metrics = {}
    train_metrics['acc'] = []
    train_metrics['loss'] = []

    val_metrics = {}
    val_metrics['acc'] = []
    val_metrics['loss'] = []

    loss_function = choosen_loss_function()
    optimizer = optimizer(params=model_nn.parameters(), lr=lr)

    best_score = 0
    best_model = None

    for epoch in tqdm.tqdm(range(epochs), desc="Training"):
        # tqdm na epoch
        numb = 0
        acc_epoch = 0
        loss_epoch = 0

        # dropout
        model_nn.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            X = X.squeeze(0).unsqueeze(1)

            output = model_nn(X)

            output = output.squeeze()

            loss_value = loss_function(output, y)


            y_pred = tensor_to_rounded_list(output)
            y_true = y

            numb_acc = count_similiarity(y_pred, y_true)

            loss_value.backward()
            optimizer.step()

            acc_epoch += numb_acc
            numb += len(y.float())
            loss_epoch += loss_value.item()

        train_metrics['acc'].append(acc_epoch / numb)
        train_metrics['loss'].append(loss_epoch / numb)

        acc_epoch = 0
        loss_epoch = 0

        numb2 = 0
        model_nn.eval()
        with torch.no_grad():
            for X, y in val_loader:

                X = X.squeeze(0).unsqueeze(1)

                output = model_nn(X)

                output = output.squeeze()

                loss_value = loss_function(output, y)

                y_pred = tensor_to_rounded_list(output)
                y_true = y

                acc = count_similiarity(y_pred, y_true)

                numb2 += len(y.float())
                acc_epoch += acc
                loss_epoch += loss_value.item()

        val_metrics['acc'].append(acc_epoch / numb2)
        val_metrics['loss'].append(loss_epoch / numb2)

        tqdm.tqdm.write(f"Epoch {epoch + 1}/{epochs} - "
                   f"Train Accuracy: {train_metrics['acc'][epoch]:.4f}, "
                   f"Train Loss: {train_metrics['loss'][epoch]:.4f},"
                   f"Val Accuracy: {val_metrics['acc'][epoch]:.4f},"
                   f"Val Loss: {val_metrics['loss'][epoch]:.4f}")

        if acc_epoch / numb2 > best_score:
            best_score = acc_epoch / numb2
            print('best score: ', best_score)


    best_model = model_nn.load_state_dict(torch.load(f"{model_name}.pt"))

    return train_metrics, val_metrics, best_model


