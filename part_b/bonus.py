import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from dataset import generate_amc_dataset
from features import extract_features
from model import AMCModel

# 1. Generate the new dataset with 512 symbols
print("Generating new dataset with n_symbols = 512...")
X_raw_512, y_512, snrs_512 = generate_amc_dataset(n_symbols=512)

# 2. Extract Features
print("Extracting features...")
X_features_512 = extract_features(X_raw_512)

# 3. Stratified Split and Scaling
stratify_labels_512 = [f"{mod}_{snr}" for mod, snr in zip(y_512, snrs_512)]
X_train_512, X_test_512, y_train_512, y_test_512, snrs_train_512, snrs_test_512 = train_test_split(
    X_features_512, y_512, snrs_512, test_size=0.2, stratify=stratify_labels_512, random_state=42
)

scaler_512 = StandardScaler()
X_train_scaled_512 = scaler_512.fit_transform(X_train_512)
X_test_scaled_512 = scaler_512.transform(X_test_512)

train_data_512 = TensorDataset(torch.FloatTensor(X_train_scaled_512), torch.LongTensor(y_train_512))
train_loader_512 = DataLoader(train_data_512, batch_size=64, shuffle=True)

# 4. Train the New Model
model_512 = AMCModel(input_dim=X_train_512.shape[1], num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_512.parameters(), lr=0.001)

epochs = 50
print("Training new model...")
for epoch in range(epochs):
    model_512.train()
    for inputs, labels in train_loader_512:
        optimizer.zero_grad()
        loss = criterion(model_512(inputs), labels)
        loss.backward()
        optimizer.step()

print("Training complete.")