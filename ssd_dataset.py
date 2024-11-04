import torch.nn.utils.spectral_norm as spectral_norm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import func_utilts as utils
import general_utils
import csv
import pickle
from collections import defaultdict
import os
from PIL import Image
import torchvision.transforms as T


def process_annotations(csv_file, images_dir):
    label_to_id = {}  # Słownik do mapowania nazw etykiet na liczby
    formatted_data = []  # Lista do przechowywania przetworzonych danych
    label_counter = 1  # Numeracja etykiet od 1; 0 jest zarezerwowane dla tła

    # Struktura przechowująca dane dla każdego obrazu
    image_annotations = defaultdict(lambda: {"boxes": [], "labels": []})

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_filename = row['filename']
            class_label = row['class']

            # Przypisz nowy numer etykiety, jeśli jeszcze nie istnieje
            if class_label not in label_to_id:
                label_to_id[class_label] = label_counter
                label_counter += 1

            # Pobierz współrzędne prostokąta otaczającego
            box = [
                int(row['xmin']),
                int(row['ymin']),
                int(row['xmax']),
                int(row['ymax'])
            ]

            # Dodaj dane do struktury dla danego obrazu
            image_annotations[img_filename]["boxes"].append(box)
            image_annotations[img_filename]["labels"].append(label_to_id[class_label])

    # Formatowanie danych (image, boxes, labels) dla każdego obrazu
    for img_filename, annotations in image_annotations.items():
        img_path = os.path.join(images_dir, img_filename)
        img = Image.open(img_path).convert("L")  # Wczytanie obrazu i konwersja do RGB
        img = np.array(img)  # Konwersja obrazu do formatu ndarray
        # print("Wymiary zdjęcia: ", img.shape)
        formatted_data.append((img, annotations["boxes"], annotations["labels"]))

    return formatted_data, label_to_id
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, device):
        super(MyDataset, self).__init__()
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)  # Ładowanie listy krotek (img, boxes, labels)
        self.device = device
        self.img_size = self.data[0][0].shape[-1]  # Rozmiar zdjęcia zakładany jako rozmiar z ostatniego wymiaru

    def __len__(self):
        return len(self.data)  # Liczba krotek w danych

    def __getitem__(self, idx):
        img, boxes, labels = self.data[idx]  # Rozpakowanie krotki na img, boxes, labels
        img = torch.tensor(img, dtype=torch.float).to(self.device)
        # print("shape: ", img.shape)
        img = (img - img.mean()) / img.std() + 0.5  # Normalizacja
        boxes = torch.tensor(boxes, dtype=torch.float32).to(self.device) / self.img_size  # [n_objects, 4]
        labels = torch.tensor(labels, dtype=torch.int64).to(self.device)  # Użycie typu int64 dla etykiet
        return img, boxes, labels

    def collate_fn(self, batch):
        imgs = []
        boxes = []
        labels = []
        for b in batch:
            imgs.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
        imgs = torch.stack(imgs)
        return imgs, boxes, labels

"""
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, device):
        super(MyDataset, self).__init__()
        with open(file_path, 'rb') as f:  # wczytanie danych za pomocą pickle
            self.data = pickle.load(f)  # (img, boxes, labels)

        self.img_size = T.ToTensor()(self.data[0][0]).shape[-1]
        self.device = device

    def __len__(self):
        return len(self.data[2])

    def __getitem__(self, idx):
        img, boxes, labels = self.data[0][idx], self.data[1][idx], self.data[2][idx]

        # Konwersja img na ndarray
        img = np.array(img)  # img w formacie [h, w] lub [h, w, c]
        if img.ndim == 2:  # Jeśli obraz jest w skali szarości, dodaj wymiar kanału
            img = np.expand_dims(img, axis=0)  # img ma teraz wymiary [c=1, h, w]
        elif img.ndim == 3:  # Jeśli obraz jest kolorowy (RGB), zmień układ na [c, h, w]
            img = np.transpose(img, (2, 0, 1))

        # Konwersja ndarray na tensor
        img = torch.tensor(img, dtype=torch.float32).to(self.device)

        img = (img - img.mean()) / img.std() + 0.5
        boxes = torch.tensor(boxes, dtype=torch.float32).to(self.device) / self.img_size  # [n_objects, 4]
        labels = torch.tensor(labels, dtype=torch.int64).to(self.device)  # [n_objects]
        return img, boxes, labels

    def collate_fn(self, batch):
        img = []
        boxes = []
        labels = []
        for b in batch:
            img.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
        img = torch.stack(img)
        return img, boxes, labels
"""