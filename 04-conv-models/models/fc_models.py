import torch.nn as nn
import torch.nn.functional as F

class FCModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.2):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for i, size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = size
            
        self.fc_layers = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return self.output(x)