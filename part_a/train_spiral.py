import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from layers import DenseLayer, ReLU, SoftmaxCrossEntropy
from optimizers import SGD, RMSProp, Adam

def generate_data(N=100, K=3, seed=42, noise=0.05):
    np.random.seed(seed)
    X = np.zeros((N*K, 2))
    y = np.zeros(N*K, dtype='uint8')
    for j in range(K):
        ix = range(N*j, N*(j+1))
        r = np.linspace(0.0, 1, N)
        theta = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*noise
        X[ix] = np.c_[r*np.sin(theta), r*np.cos(theta)]
        y[ix] = j
    return X, y

def train_model(hidden_units, opt_name, X, y, y_one_hot, epochs=1000):
    layer1 = DenseLayer(2, hidden_units)
    relu1 = ReLU()
    layer2 = DenseLayer(hidden_units, 3)
    loss_fn = SoftmaxCrossEntropy()
    
    layers = [layer1, layer2]
    if opt_name == 'SGD':
        opt = SGD(layers, lr=0.1)
    elif opt_name == 'RMSProp':
        opt = RMSProp(layers, lr=0.01)
    else:
        opt = Adam(layers, lr=0.01)
        
    losses, accs = [], []
    for epoch in range(epochs):
        z1 = layer1.forward(X)
        a1 = relu1.forward(z1)
        z2 = layer2.forward(a1)
        loss = loss_fn.forward(z2, y_one_hot)
        
        dz2 = loss_fn.backward()
        da1 = layer2.backward(dz2)
        dz1 = relu1.backward(da1)
        _ = layer1.backward(dz1)
        
        opt.update()
        
        preds = np.argmax(loss_fn.y_hat, axis=1)
        acc = np.mean(preds == y)
        losses.append(loss)
        accs.append(acc)
        
    return layers, losses, accs

def plot_decision_boundary(X, y, layers, title, filename):
    layer1, layer2 = layers
    relu1 = ReLU()
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    z1 = layer1.forward(grid)
    a1 = relu1.forward(z1)
    z2 = layer2.forward(a1)
    Z = np.argmax(z2, axis=1).reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors='k')
    plt.title(title)
    plt.savefig(f"../results/figures/{filename}")
    plt.close()

if __name__ == "__main__":
    os.makedirs("../results/figures", exist_ok=True)
    os.makedirs("../results/tables", exist_ok=True)

    X, y = generate_data()
    y_one_hot = np.zeros((y.size, 3))
    y_one_hot[np.arange(y.size), y] = 1

    results_table = []

    # Train and compare optimizers[cite: 1]
    plt.figure(figsize=(12, 5))
    for opt in ['SGD', 'RMSProp', 'Adam']:
        print(f"Training 64-neuron model with {opt}...")
        layers, losses, accs = train_model(64, opt, X, y, y_one_hot)
        results_table.append({"Optimizer": opt, "Hidden Units": 64, "Final Accuracy": accs[-1], "Final Loss": losses[-1]})
        
        plt.subplot(1, 2, 1)
        plt.plot(losses, label=opt)
        plt.subplot(1, 2, 2)
        plt.plot(accs, label=opt)
        
        # Save decision boundary for best model (Adam)
        if opt == 'Adam':
            plot_decision_boundary(X, y, layers, f"Decision Boundary (64-Neuron, {opt})", "decision_boundary_64.png")

    plt.subplot(1, 2, 1)
    plt.title("Loss vs Epoch"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy vs Epoch"); plt.legend()
    plt.savefig("../results/figures/optimizer_comparison.png")
    plt.close()

    # Train 3-neuron network[cite: 1]
    print("Training 3-neuron model with Adam...")
    layers_3, losses_3, accs_3 = train_model(3, 'Adam', X, y, y_one_hot)
    plot_decision_boundary(X, y, layers_3, "Decision Boundary (3-Neuron, Adam)", "decision_boundary_3.png")
    results_table.append({"Optimizer": "Adam", "Hidden Units": 3, "Final Accuracy": accs_3[-1], "Final Loss": losses_3[-1]})

    # Save to Table
    df = pd.DataFrame(results_table)
    df.to_csv("../results/tables/part_a_accuracies.csv", index=False)
    print("Part A training complete. Figures and Tables saved successfully.")