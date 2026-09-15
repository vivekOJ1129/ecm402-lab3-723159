import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset

from dataset import generate_amc_dataset
from features import extract_features
from model import AMCModel

# Ensure output directories exist before running
os.makedirs("../results/figures", exist_ok=True)
os.makedirs("../results/tables", exist_ok=True)

# 1. Generate Dataset
print("Generating dataset...")
X_raw, y, snrs = generate_amc_dataset()

# 2. Extract Features
print("Extracting features...")
X_features = extract_features(X_raw)

# 3. Stratified Split (by both modulation and SNR)[cite: 2]
# Combine y and snrs to create a unique class for stratification
stratify_labels = [f"{mod}_{snr}" for mod, snr in zip(y, snrs)]

X_train, X_test, y_train, y_test, snrs_train, snrs_test = train_test_split(
    X_features, y, snrs, test_size=0.2, stratify=stratify_labels, random_state=42
)

# 4. Standardize Features using training-set statistics[cite: 2]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch Tensors
train_data = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train))
test_data = TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test))

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# 5. Initialize Model, Loss, and Optimizer
model = AMCModel(input_dim=X_train.shape[1], num_classes=5)
criterion = nn.CrossEntropyLoss() # PyTorch's fused loss[cite: 2]
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Training Loop
epochs = 50
print("Starting training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

print("Training complete. Ready for evaluation (Problem 8)[cite: 2].")

# ---------------------------------------------------------
# Problem 8(a): Accuracy vs. SNR Plot
# ---------------------------------------------------------
# Ensure the model is in evaluation mode
model.eval()

unique_snrs = np.unique(snrs_test)
snr_accuracies = []

with torch.no_grad():
    for snr in unique_snrs:
        # Isolate test samples for the current SNR
        mask = (snrs_test == snr)
        X_snr = torch.FloatTensor(X_test_scaled[mask])
        y_snr = y_test[mask]
        
        # Get predictions
        outputs = model(X_snr)
        _, predicted = torch.max(outputs.data, 1)
        
        # Calculate accuracy
        acc = (predicted.numpy() == y_snr).mean()
        snr_accuracies.append(acc)

# Plotting Accuracy vs SNR
plt.figure(figsize=(8, 6))
plt.plot(unique_snrs, snr_accuracies, marker='o', linestyle='-', linewidth=2)
plt.title("Classification Accuracy vs. SNR")
plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.ylim(0, 1.05)
# Corrected save path
plt.savefig("../results/figures/accuracy_vs_snr.png")
plt.close() # Close plot to prevent it from overlapping with the next ones
print("Saved accuracy_vs_snr.png")

# ---------------------------------------------------------
# Problem 8(b): Confusion Matrices at Low and High SNR
# ---------------------------------------------------------
MODULATIONS = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]

def plot_cm(snr_mask, title, filename):
    X_sub = torch.FloatTensor(X_test_scaled[snr_mask])
    y_sub = y_test[snr_mask]
    
    with torch.no_grad():
        outputs = model(X_sub)
        _, predicted = torch.max(outputs.data, 1)
        
    cm = confusion_matrix(y_sub, predicted.numpy())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=MODULATIONS)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title(title)
    # Corrected save path
    plt.savefig(f"../results/figures/{filename}")
    plt.close()
    print(f"Saved {filename}")

# Low-SNR condition: <= 0 dB[cite: 2]
low_snr_mask = (snrs_test <= 0)
plot_cm(low_snr_mask, "Confusion Matrix (Low SNR <= 0 dB)", "cm_low_snr.png")

# High-SNR condition: >= +10 dB[cite: 2]
high_snr_mask = (snrs_test >= 10)
plot_cm(high_snr_mask, "Confusion Matrix (High SNR >= +10 dB)", "cm_high_snr.png")

# ---------------------------------------------------------
# Save SNR Accuracies to Table
# ---------------------------------------------------------
df_b = pd.DataFrame({
    "SNR (dB)": unique_snrs,
    "Accuracy": snr_accuracies
})
df_b.to_csv("../results/tables/part_b_snr_accuracies.csv", index=False)
print("Saved part_b_snr_accuracies.csv")