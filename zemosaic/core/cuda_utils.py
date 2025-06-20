import os
import subprocess


def enforce_nvidia_gpu():
    """Force usage of the first NVIDIA GPU if available."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        index = output.splitlines()[0].strip()
        if index:
            os.environ["CUDA_VISIBLE_DEVICES"] = index
            return True
    except Exception:
        pass
    return False
