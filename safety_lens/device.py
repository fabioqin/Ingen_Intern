"""Record actual hardware availability, including Apple MPS."""

import platform

import torch


def device_info():
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_devices": [
            {
                "name": torch.cuda.get_device_name(i),
                "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(torch.cuda.device_count())
        ],
        "mps_available": torch.backends.mps.is_available(),
        "training_device": "cpu",
    }
