# Copyright (C) 2024 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between
# Intel Corporation and you.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from openfl.federated import PyTorchDataLoader


class AdmissionDataLoader(PyTorchDataLoader):
    def __init__(self, data_path=None, batch_size=32, **kwargs):
        super().__init__(batch_size, **kwargs)

        self.feature_shape = [14]  # Set based on your input feature dimension
        self.num_classes = 2       # Binary classification (alive/dead)

        if data_path is None:
            return

        # Load the data
        X_train, y_train, X_valid, y_valid = load_dataset(data_path)

        # Convert to torch tensors
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
        self.X_valid = torch.tensor(X_valid, dtype=torch.float32)
        self.y_valid = torch.tensor(y_valid.reshape(-1, 1), dtype=torch.float32)

    def get_feature_shape(self):
        return self.feature_shape

    def get_num_classes(self):
        return self.num_classes


def load_dataset(data_path):
    df = pd.read_csv(data_path)

    # Categorical columns to encode
    categorical_cols = [
        'admission_location_mapped', 'discharge_location_mapped',
        'admission_type_retained', 'admittime_generalized', 'dischtime_generalized',
        'marital_status', 'ethnicity', 'religion', 'language', 'insurance', 'diagnosis'
    ]
    df[categorical_cols] = df[categorical_cols].apply(lambda col: LabelEncoder().fit_transform(col.astype(str)))

    # Features and labels
    X = df.drop(columns=['hospital_expire_flag', 'row_id', 'subject_id', 'hadm_id'])
    y = df['hospital_expire_flag'].astype('float32')

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train/validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, y_train.values, X_valid, y_valid.values
