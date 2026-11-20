import torch


def manual_linear_backward(inputs, weights, grad_output):
    grad_inputs = grad_output @ weights
    grad_weights = grad_output.transpose(0, 1) @ inputs
    grad_bias = grad_output.sum(dim=0)
    return grad_inputs, grad_weights, grad_bias


def run_gradcheck():
    inputs = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    weights = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
    bias = torch.randn(4, dtype=torch.double, requires_grad=True)

    def operation(x, weight, offset):
        return x @ weight.transpose(0, 1) + offset

    return torch.autograd.gradcheck(operation, (inputs, weights, bias))


if __name__ == "__main__":
    print(run_gradcheck())
