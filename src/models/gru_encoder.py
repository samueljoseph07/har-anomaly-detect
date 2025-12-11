import torch
import torch.nn as nn

class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
    def forward(self, x):
        _, h = self.gru(x)
        return h.squeeze(0)