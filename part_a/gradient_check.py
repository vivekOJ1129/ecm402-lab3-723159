import numpy as np
import pandas as pd
import os
from layers import DenseLayer, ReLU, SoftmaxCrossEntropy

def generate_data(N=100, K=3, seed=42):
    np.random.seed(seed)
    X = np.zeros((N*K, 2))
    y = np.zeros(N*K, dtype='uint8')
    for j in range(K):
        ix = range(N*j, N*(j+1))
        r = np.linspace(0.0, 1, N)
        theta = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.05
        X[ix] = np.c_[r*np.sin(theta), r*np.cos(theta)]
        y[ix] = j
    return X, y

if __name__ == "__main__":
    X, y = generate_data(N=10, K=3)
    y_one_hot = np.zeros((y.size, 3))
    y_one_hot[np.arange(y.size), y] = 1

    layer1 = DenseLayer(2, 64)
    relu1 = ReLU()
    layer2 = DenseLayer(64, 3)
    loss_fn = SoftmaxCrossEntropy()

    # Initial forward/backward pass to sync gradients before training[cite: 1]
    z1 = layer1.forward(X)
    a1 = relu1.forward(z1)
    z2 = layer2.forward(a1)
    _ = loss_fn.forward(z2, y_one_hot)
    dz2 = loss_fn.backward()
    da1 = layer2.backward(dz2)
    dz1 = relu1.backward(da1)
    _ = layer1.backward(dz1)

    def compute_loss(X_val, y_val):
        z1 = layer1.forward(X_val)
        a1 = relu1.forward(z1)
        z2 = layer2.forward(a1)
        return loss_fn.forward(z2, y_val)

    results = []
    
    # Check Random Weights and Biases
    for param_name, param_obj, analytic_grad in [('W', layer1.W, layer1.dL_dW), ('b', layer1.b, layer1.dL_db)]:
        idx = tuple(np.random.randint(0, d) for d in param_obj.shape)
        original_val = param_obj[idx]
        epsilon = 1e-5
        
        param_obj[idx] = original_val + epsilon
        loss_plus = compute_loss(X, y_one_hot)
        
        param_obj[idx] = original_val - epsilon
        loss_minus = compute_loss(X, y_one_hot)
        
        param_obj[idx] = original_val
        
        numeric_grad = (loss_plus - loss_minus) / (2 * epsilon)
        rel_error = abs(analytic_grad[idx] - numeric_grad) / (abs(analytic_grad[idx]) + abs(numeric_grad) + 1e-8)
        
        results.append({"Parameter": param_name, "Index": str(idx), "Relative Error": rel_error})

    # Save to table
    df = pd.DataFrame(results)
    os.makedirs("../results/tables", exist_ok=True)
    df.to_csv("../results/tables/gradient_check_results.csv", index=False)
    print("Gradient check completed. Saved to results/tables/gradient_check_results.csv")