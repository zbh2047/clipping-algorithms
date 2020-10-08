import torch.nn as nn
import torch

class Linear(nn.Module):
    def __init__(self, d, out=1, bias=True, weight=None):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d, out, bias=bias)
        if weight is not None:
            if isinstance(weight, torch.Tensor):
                self.linear.weight.data.copy_(weight.view(out, d))
            else:
                self.linear.weight.data *= weight / torch.norm(self.linear.weight.data)

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))