import torch
import torch.nn as nn

class GRU_gate(nn.Module):
    def __init__(self, n_features):

        super(GRU_gate, self).__init__()
        self.n_features = n_features

        """Reset Gate"""
        self.reset_gate = nn.Sequential(
            nn.Linear(2 * n_features, n_features),
            # 激活函数为 sigmoid
            nn.Sigmoid()
        )
        """Update Gate"""
        self.update_gate = nn.Sequential(
            # 线性变换就相当于矩阵乘
            nn.Linear(2 * n_features, n_features),
            nn.Sigmoid()
        )
        """The output transform"""
        self.transform = nn.Sequential(
            nn.Linear(2 * n_features, n_features),
            nn.Tanh()
        )

    def forward(self, h, h_in):
        a = torch.cat((h, h_in), 1)

        r = self.reset_gate(a)
        z = self.update_gate(a)

        joined_input = torch.cat((h, r * h_in), 1)
        h_hat = self.transform(joined_input)

        output = (1 - z) * h_in + z * h_hat
        return output