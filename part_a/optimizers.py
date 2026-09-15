import numpy as np

class SGD:
    def __init__(self, layers, lr=0.01, momentum=0.9):
        self.layers = layers
        self.lr = lr
        self.momentum = momentum
        self.v = [{'W': np.zeros_like(l.W), 'b': np.zeros_like(l.b)} for l in layers]
        
    def update(self):
        for i, layer in enumerate(self.layers):
            for param, grad in [('W', layer.dL_dW), ('b', layer.dL_db)]:
                self.v[i][param] = self.momentum * self.v[i][param] - self.lr * grad
                if param == 'W':
                    layer.W += self.v[i][param]
                else:
                    layer.b += self.v[i][param]

class RMSProp:
    def __init__(self, layers, lr=0.01, rho=0.99, eps=1e-8):
        self.layers = layers
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.s = [{'W': np.zeros_like(l.W), 'b': np.zeros_like(l.b)} for l in layers]
        
    def update(self):
        for i, layer in enumerate(self.layers):
            for param, grad in [('W', layer.dL_dW), ('b', layer.dL_db)]:
                self.s[i][param] = self.rho * self.s[i][param] + (1 - self.rho) * (grad ** 2)
                update = self.lr * grad / (np.sqrt(self.s[i][param]) + self.eps)
                if param == 'W':
                    layer.W -= update
                else:
                    layer.b -= update

class Adam:
    def __init__(self, layers, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.layers = layers
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [{'W': np.zeros_like(l.W), 'b': np.zeros_like(l.b)} for l in layers]
        self.s = [{'W': np.zeros_like(l.W), 'b': np.zeros_like(l.b)} for l in layers]
        
    def update(self):
        self.t += 1
        for i, layer in enumerate(self.layers):
            for param, grad in [('W', layer.dL_dW), ('b', layer.dL_db)]:
                self.m[i][param] = self.beta1 * self.m[i][param] + (1 - self.beta1) * grad
                self.s[i][param] = self.beta2 * self.s[i][param] + (1 - self.beta2) * (grad ** 2)
                
                m_hat = self.m[i][param] / (1 - self.beta1 ** self.t)
                s_hat = self.s[i][param] / (1 - self.beta2 ** self.t)
                
                update = self.lr * m_hat / (np.sqrt(s_hat) + self.eps)
                if param == 'W':
                    layer.W -= update
                else:
                    layer.b -= update