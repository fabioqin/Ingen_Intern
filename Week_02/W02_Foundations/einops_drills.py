import torch


def run_einops_drills(values=None):
    if values is None:
        values = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    if values.ndim != 3:
        raise ValueError("Input must have shape (batch, height, width).")

    try:
        from einops import rearrange, reduce, repeat
    except ImportError:
        flattened = values.reshape(values.shape[0], -1)
        pooled = values.mean(dim=(1, 2))
        duplicated = values.unsqueeze(1).expand(-1, 2, -1, -1)
    else:
        flattened = rearrange(values, "b h w -> b (h w)")
        pooled = reduce(values, "b h w -> b", "mean")
        duplicated = repeat(values, "b h w -> b copies h w", copies=2)

    return {
        "flattened": flattened,
        "pooled": pooled,
        "duplicated": duplicated,
    }


if __name__ == "__main__":
    for name, value in run_einops_drills().items():
        print(name, value.shape)
