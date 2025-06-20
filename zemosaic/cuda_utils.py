import os
import subprocess


def enforce_nvidia_gpu():
    """Force l'utilisation du premier GPU NVIDIA via CUDA_VISIBLE_DEVICES."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        idx = out.strip().splitlines()[0]
        if idx:
            os.environ["CUDA_VISIBLE_DEVICES"] = idx
            return True
    except Exception:
        pass
    return False

