import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
from torchvision import models
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torchvision

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import os
import json
import re
import random
from math import sqrt

from new_utilts import xy_to_cxcy, cxcy_to_xy, cxcy_to_gcxgcy, gcxgcy_to_cxcy, find_jaccard_overlap, decimate, calculate_mAP

def prepare_tsfm():

    tsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return tsfm


class SSDDateset(Dataset):
    def __init__(self, img_names, annotations_file, subset='train', transform=prepare_tsfm()):
        self.img_folder_path = f"C:/Users/izaol/fridge_food/jedzenie/{subset}/images/"
        self.annotations_file = annotations_file
        self.img_names = img_names
        self.transform = transform

        self.annotations_df = pd.read_csv(self.annotations_file)

    def __getitem__(self, idx):
        file = self.img_names[idx]

        img_path = self.img_folder_path + file
        img = Image.open(img_path)
        img = img.convert('RGB')

        file_annotations = self.annotations_df[self.annotations_df['filename'] == file]

        boxes, labels = [], []
        for _, row in file_annotations.iterrows():
            # Pobierz klasę (dodaj 1, aby uwzględnić tło)
            label = int(row['class']) + 1
            labels.append(label)

            # Pobierz współrzędne i zamień na (x_min, y_min, x_max, y_max)
            x_min = row['xmin']
            y_min = row['ymin']
            x_max = row['xmax']
            y_max = row['ymax']
            boxes.append([x_min, y_min, x_max, y_max])

        # Konwersja boxes i labels na tensory
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.transform is not None:
            img = self.transform(img)

        return img, boxes, labels

    def __len__(self):
        return len(self.img_names)

    """def show_box(self):
        image, boxes, labels = self[np.random.randint(len(self))]
        image = image.detach().cpu()
        boxes = boxes.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy().tolist()

        if self.transform is not None:
            image = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
                                        transforms.ToPILImage()])(image)
        else:
            image = transforms.ToPILImage()(image)
        width, height = image.size

        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height

        draw = ImageDraw.Draw(image)
        for box, label in zip(boxes, labels):
            draw.rectangle(xy=[tuple(box.tolist())[:2], tuple(box.tolist())[2:]])
            xmin, ymin = tuple(box.tolist())[:2]
            draw.text(xy=[xmin, ymin], text=id_to_class[int(label)])

        return image"""

    def cxcy_to_xy(self, cxcy):
        return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2), cxcy[:, :2] + (cxcy[:, 2:] / 2)], dim=1)

    @staticmethod
    def collate_fn(batch):
        images = [item[0] for item in batch]
        boxes = [item[1] for item in batch]
        labels = [item[2] for item in batch]

        # Stakuj obrazy (bo mają ten sam wymiar), a boxes i labels zostaw jako listy
        images = torch.stack(images, dim=0)

        return images, boxes, labels