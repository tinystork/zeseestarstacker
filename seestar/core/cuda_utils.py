import os
import subprocess


def enforce_nvidia_gpu():
    """Force l'utilisation du premier GPU NVIDIA disponible via CUDA_VISIBLE_DEVICES.

    Returns:
        bool: True si la variable a été définie avec succès, False sinon.
    """
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        first_index = output.strip().splitlines()[0]
        if first_index:
            os.environ["CUDA_VISIBLE_DEVICES"] = first_index
            return True
    except Exception:
        pass
    return False

