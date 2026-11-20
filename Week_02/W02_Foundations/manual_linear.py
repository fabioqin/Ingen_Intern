from math import sqrt

import torch
from torch import nn


class ManualLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("Feature dimensions must be positive.")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension {self.in_features}, got {inputs.shape[-1]}."
            )
        outputs = inputs @ self.weight.transpose(0, 1)
        if self.bias is not None:
            outputs = outputs + self.bias
        return outputs


if __name__ == "__main__":
    layer = ManualLinear(4, 3)
    sample = torch.randn(2, 4)
    print(layer(sample))
