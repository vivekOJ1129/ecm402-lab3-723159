# ECM-402 Lab 3: Feedforward Neural Networks (From Scratch & PyTorch)

**Course:** ECM-402: AI for Signal Processing (Ist Semester 2026-27)  
**Institute:** National Institute of Technology, Andhra Pradesh  
**Instructor:** Dr. V. Prakash Singh  

## Overview
This repository contains the implementation for Lab 3, divided into two distinct parts:
*   **Part A (From Scratch):** A feedforward neural network built entirely from scratch using NumPy. It includes implementations for dense layers, ReLU, fused Softmax-CrossEntropy, and custom optimizers (SGD with momentum, RMSProp, Adam). The network is trained on a 3-class non-linear spiral dataset.
*   **Part B (PyTorch):** Application of a Multi-Layer Perceptron (MLP) using PyTorch for Automatic Modulation Classification (AMC). The model classifies 5 digital modulation schemes (BPSK, QPSK, 8PSK, 16QAM, 64QAM) from synthetic complex baseband signals across varying Signal-to-Noise Ratios (SNRs).

## Repository Structure

```text
ecm402_lab3_<your_rollnumber>/
│
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── report.pdf                 # Final lab report with derivations and analysis
│
├── part_a/                    # NumPy-only MLP Implementation
│   ├── layers.py              # DenseLayer, ReLU, SoftmaxCrossEntropy
│   ├── optimizers.py          # SGD, RMSProp, Adam
│   ├── gradient_check.py      # Finite-difference gradient verification
│   └── train_spiral.py        # Training loop and decision boundary plotting
│
├── part_b/                    # PyTorch AMC Implementation
│   ├── dataset.py             # AMC synthetic signal generation
│   ├── features.py            # Amplitude, phase, and cumulant feature extraction
│   ├── model.py               # PyTorch MLP architecture
│   ├── train_amc.py           # Training loop, evaluation, and confusion matrices
│   └── bonus.py               # (Optional) Hypothesis testing with n_symbols=512
│
├── results/                   # Auto-generated outputs
│   ├── figures/               # Saved plots (boundaries, SNR accuracy, CMs)
│   └── tables/                # Saved CSV data (gradients, final accuracies)
│
└── notebooks/                 # Exploratory Jupyter notebooks
