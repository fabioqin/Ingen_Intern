import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from Week_02.W02_Foundations.autograd_gradcheck import (
    manual_linear_backward,
    run_gradcheck,
)
from Week_02.W02_Foundations.einops_drills import run_einops_drills
from Week_02.W02_Foundations.manual_linear import ManualLinear
from Week_02.W02_Foundations.mnist_experiment import (
    MNISTMLP,
    evaluate,
    train_epoch,
)
from Week_02.W02_Foundations.softmax_cross_entropy import (
    cross_entropy,
    stable_softmax,
)
from Week_02.W02_Foundations.tensor_drills import run_tensor_drills
from Week_02.W02_Foundations.two_layer_mlp import TwoLayerMLP


def test_tensor_drills():
    result = run_tensor_drills()
    assert result["values"].shape == (3, 4)
    assert result["transposed"].shape == (4, 3)
    assert result["matrix_product"].shape == (3, 2)
    assert result["selected_value"] == 6.0


def test_manual_linear_forward_and_backward():
    torch.manual_seed(7)
    layer = ManualLinear(3, 2).double()
    inputs = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
    outputs = layer(inputs)
    grad_output = torch.randn_like(outputs)
    outputs.backward(grad_output)
    grad_inputs, grad_weights, grad_bias = manual_linear_backward(
        inputs.detach(), layer.weight.detach(), grad_output
    )
    assert outputs.shape == (4, 2)
    assert torch.allclose(inputs.grad, grad_inputs)
    assert torch.allclose(layer.weight.grad, grad_weights)
    assert torch.allclose(layer.bias.grad, grad_bias)


def test_softmax_and_cross_entropy():
    logits = torch.tensor([[10000.0, 9999.0, -10000.0], [1.0, 2.0, 3.0]])
    targets = torch.tensor([0, 2])
    probabilities = stable_softmax(logits)
    loss = cross_entropy(logits, targets)
    assert torch.isfinite(probabilities).all()
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.allclose(loss, functional.cross_entropy(logits, targets))


def test_two_layer_mlp():
    model = TwoLayerMLP(4, 12, 3)
    inputs = torch.randn(5, 4)
    outputs = model(inputs)
    outputs.square().mean().backward()
    assert outputs.shape == (5, 3)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_autograd_gradcheck():
    assert run_gradcheck()


def test_einops_drills():
    result = run_einops_drills(torch.randn(2, 3, 4))
    assert result["flattened"].shape == (2, 12)
    assert result["pooled"].shape == (2,)
    assert result["duplicated"].shape == (2, 2, 3, 4)


def test_mnist_model_forward():
    model = MNISTMLP()
    outputs = model(torch.randn(6, 1, 28, 28))
    assert outputs.shape == (6, 10)


def test_mnist_training_and_evaluation():
    torch.manual_seed(11)
    images = torch.rand(24, 1, 28, 28)
    labels = torch.randint(0, 10, (24,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)
    model = MNISTMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loss = train_epoch(model, loader, optimizer)
    test_loss, accuracy = evaluate(model, loader)
    assert torch.isfinite(torch.tensor(train_loss))
    assert torch.isfinite(torch.tensor(test_loss))
    assert 0.0 <= accuracy <= 1.0


def test_two_layer_backprop_matches_chain_rule():
    torch.manual_seed(42)
    model = TwoLayerMLP(3, 4, 2).double()
    inputs = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
    upstream = torch.randn(5, 2, dtype=torch.double)
    model(inputs).backward(upstream)
    with torch.no_grad():
        preactivation = model.hidden(inputs)
        hidden = preactivation.relu()
        grad_hidden, grad_w2, grad_b2 = manual_linear_backward(
            hidden, model.output.weight, upstream
        )
        grad_x, grad_w1, grad_b1 = manual_linear_backward(
            inputs, model.hidden.weight, grad_hidden * (preactivation > 0)
        )
    for actual, expected in zip(
        [inputs.grad, *(parameter.grad for parameter in model.parameters())],
        [grad_x, grad_w1, grad_b1, grad_w2, grad_b2],
    ):
        torch.testing.assert_close(actual, expected)
