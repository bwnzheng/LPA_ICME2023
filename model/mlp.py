import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, hidden_dim=256) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(28*28, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, 10),
        ) 
        
    def forward(self, x):
        return self.net(x);