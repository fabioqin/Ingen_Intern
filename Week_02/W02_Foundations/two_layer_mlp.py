import torch
from torch import nn

from Week_02.W02_Foundations.manual_linear import ManualLinear


class TwoLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden = ManualLinear(input_size, hidden_size)
        self.output = ManualLinear(hidden_size, output_size)

    def forward(self, inputs):
        hidden = torch.relu(self.hidden(inputs))
        return self.output(hidden)


if __name__ == "__main__":
    model = TwoLayerMLP(4, 16, 3)
    print(model(torch.randn(5, 4)).shape)
