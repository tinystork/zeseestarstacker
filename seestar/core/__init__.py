"""
Package core pour Seestar - fournit les fonctionnalités de base pour le traitement des images astronomiques.
(stacking.py a été retiré car remplacé par queue_manager.py)
"""

# Compatibilité NumPy 2.x : certaines versions de ``photutils`` s'attendent à
# ``numpy.lib.index_tricks.index_exp``. Si ce module n'est plus exposé,
# on crée un équivalent minimal utilisant ``numpy.index_exp``.
import sys
import types
import numpy as np

try:  # pragma: no cover - si la compatibilité est déjà présente
    from numpy.lib import index_tricks as _idx_mod  # noqa: F401
except Exception:  # pragma: no cover - pour environnements NumPy 2+
    stub = types.ModuleType("numpy.lib.index_tricks")
    stub.index_exp = np.index_exp
    sys.modules.setdefault("numpy.lib.index_tricks", stub)

from .image_processing import (
    load_and_validate_fits,
    debayer_image,
    save_fits_image,
    save_preview_image,
)

from .hot_pixels import detect_and_correct_hot_pixels
from .utils import estimate_batch_size, apply_denoise, check_cuda, check_cupy_cuda # Ajout des checks CUDA/CuPy
from .alignment import SeestarAligner # C'est l'aligneur basé sur astroalign
from .weights import (
    _calculate_image_weights_noise_variance,
    _calculate_image_weights_noise_fwhm,
)
from .normalization import (
    _normalize_images_linear_fit,
    _normalize_images_sky_mean,
)
from .stack_methods import (
    _stack_mean,
    _stack_median,
    _stack_kappa_sigma,
    _stack_linear_fit_clip,
)
from .incremental_reprojection import reproject_and_combine
from .reprojection import resolve_all_wcs
from .reprojection_utils import (
    collect_headers,
    compute_final_output_grid,
    standardize_wcs,
)
from .simple_stacker import create_master_tile as create_master_tile_simple

# Liste initiale des éléments à exporter
__all__ = [
    'load_and_validate_fits',
    'debayer_image',
    'detect_and_correct_hot_pixels',
    'save_fits_image',
    'save_preview_image',
    'estimate_batch_size',
    'apply_denoise',
    'check_cuda',             # Exposer la fonction de vérification CUDA
    'check_cupy_cuda',        # Exposer la fonction de vérification CuPy
    'SeestarAligner',         # L'aligneur astroalign
    '_calculate_image_weights_noise_variance',
    '_calculate_image_weights_noise_fwhm',
    '_normalize_images_linear_fit',
    '_normalize_images_sky_mean',
    '_stack_mean',
    '_stack_median',
    '_stack_kappa_sigma',
    '_stack_linear_fit_clip',
    'create_master_tile_simple',
    'reproject_and_combine',
    'resolve_all_wcs',
    'collect_headers',
    'compute_final_output_grid',
    'standardize_wcs',
    'downsample_image'
]

# Tentative d'importation du nouvel aligneur local
try:
    from .fast_aligner_module import FastSeestarAligner as SeestarLocalAligner  # noqa: F401
    print("DEBUG [core/__init__.py]: SeestarLocalAligner (FastSeestarAligner) importé avec succès.")
    __all__.append('SeestarLocalAligner') # Ajouter à __all__ SI l'import réussit
except ImportError as e_fla:
    print(f"WARN [core/__init__.py]: FastSeestarAligner non importable depuis fast_aligner_module: {e_fla}")
    # Optionnel: définir SeestarLocalAligner comme None pour que le reste du code puisse vérifier son existence
    # Cependant, si une partie du code essaie de l'utiliser sans vérifier, cela plantera.
    # Il est peut-être préférable de laisser l'ImportError se propager si c'est une dépendance critique
    # pour une fonctionnalité activée. Pour l'instant, on logue juste.
    # SeestarLocalAligner = None
except Exception as e_other_fla:
    print(f"ERREUR [core/__init__.py]: Erreur inattendue lors de l'import de FastSeestarAligner: {e_other_fla}")
    import traceback
    traceback.print_exc(limit=2)

print(f"DEBUG [core/__init__.py]: Contenu final de __all__: {__all__}")
# --- END OF FILE seestar/core/__init__.py ---
