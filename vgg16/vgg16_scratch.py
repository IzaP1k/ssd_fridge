import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
from torchvision import models
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, TensorDataset

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import os
import json
import re
import random
from math import sqrt


class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Aktualizacja: 9*9*512 = 41472
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9 * 9 * 512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# USING

num_classes = 4
num_epochs = 50
batch_size = 30
learning_rate = 0.001


def create_model(num_classes, num_epochs, batch_size, learning_rate):

    model = VGG16(num_classes)


    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

    return model, criterion, optimizer

def train_loop(model, criterion, optimizer, device, train_loader, valid_loader):

    total_step = len(train_loader)
    # training loop
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Take the Tensors onto the device
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            # print(images.shape)
            images = images.permute(0, 3, 1, 2)
            # print(images.shape)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)

                images = images.permute(0, 3, 1, 2)
                outputs = model(images)

                """print("output: ",outputs)
                print("labels: ", labels)"""

                errors_per_column = (outputs != labels).sum(dim=0)

                # Średnia różnica między przewidywaniami a etykietami w każdej kolumnie
                average_difference_per_column = torch.abs(outputs - labels).float().mean(dim=0)


                print("Liczba błędów przewidywań dla każdej kolumny:", errors_per_column)
                print("Średnia różnica między przewidywaniami a etykietami dla każdej kolumny:",
                      average_difference_per_column)
                del images, labels, outputs


    return model
def change_labels(train_annotations, valid_annotations, test_annotations):
    # Concatenate all annotations to create a unified DataFrame
    all_annotations = pd.concat([train_annotations, valid_annotations, test_annotations])

    # Map unique product names to integers (0, 1, 2, ...)
    unique_products = all_annotations['class'].unique()
    product_to_number = {product: idx for idx, product in enumerate(unique_products)}

    # Update class labels for each dataset
    for df in [train_annotations, valid_annotations, test_annotations]:
        df['class'] = df['class'].map(product_to_number)

    train_class_counts = train_annotations.pivot_table(
        index='filename',
        columns='class',
        aggfunc='size',  # Count occurrences
        fill_value=0  # Fill missing combinations with 0
    ).reset_index()

    val_class_counts = valid_annotations.pivot_table(
        index='filename',
        columns='class',
        aggfunc='size',
        fill_value=0
    ).reset_index()

    test_class_counts = test_annotations.pivot_table(
        index='filename',
        columns='class',
        aggfunc='size',
        fill_value=0
    ).reset_index()

    # Rename columns for better readability
    train_class_counts.columns = ['filename'] + [f'class_{int(col)}' for col in train_class_counts.columns[1:]]
    val_class_counts.columns = ['filename'] + [f'class_{int(col)}' for col in val_class_counts.columns[1:]]
    test_class_counts.columns = ['filename'] + [f'class_{int(col)}' for col in test_class_counts.columns[1:]]

    return train_class_counts, val_class_counts, test_class_counts

def get_data(df, path):
    vector_data = []

    for idx, row in df.iterrows():
        filename = row['filename']
        vector = row.iloc[1:].values
        image_path = os.path.join(path, filename)

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img = np.array(img)
                vector_data.append([img.copy(), vector])
        except FileNotFoundError:
            print(f"Obraz {filename} nie został znaleziony.")

    # Tworzymy nowy DataFrame z kolumnami 'image' i 'class_vector'
    vector_df = pd.DataFrame(vector_data, columns=['image', 'class_vector'])

    return vector_df

def split_tensor_data(train_data, valid_data, test_data, batch_size=10):
    X_train = np.stack(train_data['image'])
    y_train = torch.tensor(train_data['class_vector'].tolist(), dtype=torch.float)

    X_val = np.stack(valid_data['image'])
    y_val = torch.tensor(valid_data['class_vector'].tolist(), dtype=torch.float)

    X_test = torch.tensor(test_data['image'].tolist(), dtype=torch.float)
    y_test = torch.tensor(test_data['class_vector'].tolist(), dtype=torch.float)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), y_train)
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float), y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset)

    return train_loader, val_loader, test_loader
