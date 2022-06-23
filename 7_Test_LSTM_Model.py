import os
from os import walk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import seaborn as sns

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import accuracy
from sklearn.metrics import classification_report, confusion_matrix

cwd = os.getcwd()

test_files_path = cwd + '\\Final Data\\Test\\'

test_filenames = next(walk(test_files_path), (None, None, []))[2]


def data_list(files_path, filenames):
    ataxia_data_list = []
    normal_data_list = []

    num = int(len(filenames) / 2)

    for i in range(num):
        ataxia_data = pd.read_csv(files_path + filenames[i])
        ataxia_data_list.append(ataxia_data)
        normal_data = pd.read_csv(files_path + filenames[i + num])
        normal_data_list.append(normal_data)

    return ataxia_data_list, normal_data_list


def dataset(ataxia_data_list, normal_data_list):
    sequences = []

    for ataxia_dataset, normal_dataset in zip(ataxia_data_list, normal_data_list):

        ataxia_dataset.drop(ataxia_dataset.tail(ataxia_dataset.shape[0] % 256).index, inplace=True)

        series_id_ataxia = []
        ID_ataxia_train = 0
        for i in range(int(ataxia_dataset.shape[0] / 256)):
            for j in range(256):
                series_id_ataxia.append(ID_ataxia_train)
            ID_ataxia_train += 1

        predictions_ataxia = {'SeriesID': series_id_ataxia,
                              'GaitType': ['Ataxic Gait' for _ in range(ataxia_dataset.shape[0])]}

        normal_dataset.drop(normal_dataset.tail(normal_dataset.shape[0] % 256).index, inplace=True)

        series_id_normal = []
        ID_normal = int(ataxia_dataset.shape[0] / 256)

        for i in range(int(normal_dataset.shape[0] / 256)):
            for j in range(256):
                series_id_normal.append(ID_normal)
            ID_normal += 1

        predictions_normal = {'SeriesID': series_id_normal,
                              'GaitType': ['Normal Gait' for _ in range(normal_dataset.shape[0])]}

        predict_ataxia = pd.DataFrame(predictions_ataxia)
        predict_normal = pd.DataFrame(predictions_normal)

        x = pd.concat((ataxia_dataset, normal_dataset), ignore_index=True)
        y = pd.concat((predict_ataxia, predict_normal), ignore_index=True)

        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(y.GaitType)

        x.insert(0, 'SeriesID', y.SeriesID)
        y.insert(2, 'Label', encoded_labels)

        FeatureColumns = x.columns.tolist()[1:]

        Sequences = []
        for series_id, group in x.groupby('SeriesID'):
            sequence_features = group[FeatureColumns]
            label = y[y.SeriesID == series_id].iloc[0].Label
            Sequences.append((sequence_features, label))

        for seq in Sequences:
            sequences.append(seq)

    return sequences, label_encoder, FeatureColumns


test_sequences = dataset(data_list(test_files_path, test_filenames)[0],
                         data_list(test_files_path, test_filenames)[1]
                         )[0]
label_encoder = dataset(data_list(test_files_path, test_filenames)[0],
                        data_list(test_files_path, test_filenames)[1]
                        )[1]
FeatureColumns = dataset(data_list(test_files_path, test_filenames)[0],
                         data_list(test_files_path, test_filenames)[1]
                         )[2]

normal_labels_test = []
ataxia_labels_test = []
for i in test_sequences:
    if i[1] == 0:
        ataxia_labels_test.append(i[1])
    else:
        normal_labels_test.append(i[1])

print(len(ataxia_labels_test), len(normal_labels_test))


class GaitDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        return dict(
            sequence=torch.Tensor(sequence.to_numpy()),
            label=torch.tensor(label).long()
        )


class GaitDataModule(pl.LightningDataModule):
    def __init__(self, test_sequences, batch_size):
        super().__init__()
        self.test_sequences = test_sequences
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.test_dataset = GaitDataset(self.test_sequences)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )


N_EPOCHS = 150
BATCH_SIZE = 64

data_module = GaitDataModule(test_sequences, BATCH_SIZE)


class SequenceModel(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden=256, n_layers=4):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.75
        )
        self.classifier = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)

        out = hidden[-1]
        return self.classifier(out)


class GaitPredictor(pl.LightningModule):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.model = SequenceModel(n_features, n_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = accuracy(predictions, labels)

        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_accuracy', step_accuracy, prog_bar=True, logger=True)

        return {'loss': loss, 'accuracy': step_accuracy}

    def validation_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = accuracy(predictions, labels)

        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_accuracy', step_accuracy, prog_bar=True, logger=True)

        return {'loss': loss, 'accuracy': step_accuracy}

    def test_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = accuracy(predictions, labels)

        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_accuracy', step_accuracy, prog_bar=True, logger=True)

        return {'loss': loss, 'accuracy': step_accuracy}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.00001)


model = GaitPredictor(
    n_features=len(FeatureColumns),
    n_classes=len(label_encoder.classes_)
)

PATH = './Final_LSTM_Model/gait_net_best_model.pth'

model.load_state_dict(torch.load(PATH))
model.eval()

test_dataset = GaitDataset(test_sequences)

predictions = []
labels = []

for item in tqdm(test_dataset):
    sequence = item['sequence']
    label = item['label']

    _, output = model(sequence.unsqueeze(dim=0))
    prediction = torch.argmax(output, dim=1)
    predictions.append(prediction.item())
    labels.append(label.item())

print(
    classification_report(labels, predictions, target_names=label_encoder.classes_)
)


def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=0, ha='right')
    plt.ylabel('True Gait Type')
    plt.xlabel('Predicted Gait Type')


cm = confusion_matrix(labels, predictions)
df_cm = pd.DataFrame(
    cm, index=label_encoder.classes_,
    columns=label_encoder.classes_
)

show_confusion_matrix(df_cm)

