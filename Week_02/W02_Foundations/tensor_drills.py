import torch


def run_tensor_drills():
    values = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    weights = torch.ones((4, 2), dtype=values.dtype)
    return {
        "values": values,
        "transposed": values.transpose(0, 1),
        "matrix_product": values @ weights,
        "selected_value": values[1, 2].item(),
    }


if __name__ == "__main__":
    for name, value in run_tensor_drills().items():
        print(name, value)
