"""Usage: uv run python -m scripts.check_gpu."""

import json
from pathlib import Path

from safety_lens.device import device_info

if __name__ == "__main__":
    result = json.dumps(device_info(), indent=2)
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/device.json").write_text(result + "\n")
    print(result)
