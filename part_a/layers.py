import numpy as np

class DenseLayer:
    def __init__(self, n_in, n_out):
        # He initialization for ReLU networks
        self.W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        self.b = np.zeros((1, n_out))
        
    def forward(self, X):
        self.X = X
        return np.dot(X, self.W) + self.b
        
    def backward(self, dL_dz):
        self.dL_dW = np.dot(self.X.T, dL_dz)
        self.dL_db = np.sum(dL_dz, axis=0, keepdims=True)
        return np.dot(dL_dz, self.W.T)

class ReLU:
    def forward(self, z):
        self.z = z
        return np.maximum(0, z)
        
    def backward(self, dL_da):
        return dL_da * (self.z > 0)

class SoftmaxCrossEntropy:
    def forward(self, z, y_true):
        self.y_true = y_true
        # Numerically stable softmax
        shifted_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shifted_z)
        self.y_hat = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        n = z.shape[0]
        y_hat_clipped = np.clip(self.y_hat, 1e-9, 1 - 1e-9)
        loss = -np.sum(y_true * np.log(y_hat_clipped)) / n
        return loss
        
    def backward(self):
        n = self.y_hat.shape[0]
        # Fused gradient simplification[cite: 1]
        return (self.y_hat - self.y_true) / n