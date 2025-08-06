# Copyright (C) 2024 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between
# Intel Corporation and you.

# Copyright (C) 2024 Intel Corporation

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from typing import Iterator, Tuple
from openfl.federated import PyTorchTaskRunner
from openfl.utilities import Metric
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.model import model_1
from src.model.model_1 import MLP  # Adjusted import to match the new structure`` 
MLP()



class AdmissionTaskRunner(PyTorchTaskRunner):
    def __init__(self, device="cpu", **kwargs):
        super().__init__(device=device, **kwargs)

        # Define model
        self.model = MLP(input_dim=14).to(device)  # 15 = number of features
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def train_(self, train_dataloader: Iterator[Tuple[np.ndarray, np.ndarray]]) -> Metric:
        self.model.train()
        total_loss = 0
        total_samples = 0

        for X_batch, y_batch in train_dataloader:
            X_batch = torch.tensor(X_batch, dtype=torch.float32).to(self.device)
            y_batch = torch.tensor(y_batch, dtype=torch.float32).to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.loss_fn(outputs, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)

        avg_loss = total_loss / total_samples
        return Metric(name="avg_loss", value=np.array(avg_loss))

    def validate_(self, validation_dataloader: Iterator[Tuple[np.ndarray, np.ndarray]]) -> Metric:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in validation_dataloader:
                X_batch = torch.tensor(X_batch, dtype=torch.float32).to(self.device)
                y_batch = torch.tensor(y_batch, dtype=torch.float32).to(self.device)

                outputs = self.model(X_batch)
                predicted = (outputs > 0.5).float()
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

        accuracy = correct / total
        return Metric(name="accuracy", value=np.array(accuracy))



