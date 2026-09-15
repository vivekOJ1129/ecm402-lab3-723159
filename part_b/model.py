import torch
import torch.nn as nn

class AMCModel(nn.Module):
    def __init__(self, input_dim=11, num_classes=5):
        super(AMCModel, self).__init__()
        # Two hidden layers of 128 units as suggested[cite: 2]
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
            # Softmax is omitted here because nn.CrossEntropyLoss expects raw logits[cite: 2]
        )

    def forward(self, x):
        return self.network(x)