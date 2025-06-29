# zemosaic_align_stack.py

import numpy as np
import traceback
import gc
import logging  # Added for logger fallback
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# dépendance Photutils
PHOTOUTILS_AVAILABLE = False
DAOStarFinder, FITSFixedWarning, CircularAperture, aperture_photometry, SigmaClip, Background2D, MedianBackground, SourceCatalog = [None]*8 # type: ignore
try:
    from astropy.stats import SigmaClip, gaussian_sigma_to_fwhm # gaussian_sigma_to_fwhm est utile
    from astropy.table import Table
    from photutils.detection import DAOStarFinder
    from photutils.aperture import CircularAperture, aperture_photometry
    from photutils.background import Background2D, MedianBackground
    from photutils.segmentation import detect_sources, SourceCatalog

# ... et _internal_logger
    # Ignorer les avertissements FITSFixedWarning de photutils si besoin
    import warnings
    from astropy.wcs import FITSFixedWarning
    warnings.filterwarnings('ignore', category=FITSFixedWarning)
    
    PHOTOUTILS_AVAILABLE = True
    # print("INFO (zemosaic_align_stack): Photutils importé.")
except ImportError:
    print("AVERT (zemosaic_align_stack): Photutils non disponible. FWHM weighting limité.")

# --- Dépendance Astroalign ---
ASTROALIGN_AVAILABLE = False
astroalign_module = None 
try:
    import astroalign as aa
    astroalign_module = aa
    ASTROALIGN_AVAILABLE = True
    # print("INFO (zemosaic_align_stack): Astroalign importé.") # Log au démarrage du worker principal
except ImportError:
    print("ERREUR CRITIQUE (zemosaic_align_stack): Astroalign non installé. Alignement impossible.")

# --- Dépendance Astropy (pour sigma_clipped_stats) ---
SIGMA_CLIP_AVAILABLE = False
sigma_clipped_stats_func = None
try:
    from astropy.stats import sigma_clipped_stats
    sigma_clipped_stats_func = sigma_clipped_stats
    SIGMA_CLIP_AVAILABLE = True
    # print("INFO (zemosaic_align_stack): Astropy.stats.sigma_clipped_stats importé.")
except ImportError:
    print("AVERT (zemosaic_align_stack): Astropy.stats non disponible. Kappa-sigma stacking limité.")

# --- Dépendance SciPy (pour Winsorize) ---
SCIPY_AVAILABLE = False
winsorize_func = None
try:
    from scipy.stats.mstats import winsorize
    winsorize_func = winsorize
    SCIPY_AVAILABLE = True
    # print("INFO (zemosaic_align_stack): Scipy.stats.mstats.winsorize importé.")
except ImportError:
    print("AVERT (zemosaic_align_stack): Scipy non disponible. Winsorized Sigma Clip non fonctionnel.")

ZEMOSAIC_UTILS_AVAILABLE_FOR_RADIAL = False
make_radial_weight_map_func = None
try:
    from zemosaic_utils import make_radial_weight_map
    make_radial_weight_map_func = make_radial_weight_map
    ZEMOSAIC_UTILS_AVAILABLE_FOR_RADIAL = True
except ImportError as e_util_rad:
        print(f"AVERT (zemosaic_align_stack): Radial weighting: Erreur import make_radial_weight_map: {e_util_rad}")


# Fallback logger for cases where progress_callback might not be available
# or for internal print-like debugging within this module if necessary.
_internal_logger = logging.getLogger("ZeMosaicAlignStackInternal")
if not _internal_logger.hasHandlers():
    _internal_logger.setLevel(logging.DEBUG)
    # Add a null handler to prevent "No handler found" warnings if not configured elsewhere
    # _internal_logger.addHandler(logging.NullHandler()) # Or configure a basic one if needed for standalone tests.



def _calculate_robust_stats_for_linear_fit(image_data_2d_float32: np.ndarray,
                                           low_percentile: float = 25.0,
                                           high_percentile: float = 90.0,
                                           progress_callback: callable = None):
    """
    Calcule des statistiques robustes (deux points de percentiles) pour une image 2D (un canal).
    Utilisé par la normalisation par ajustement linéaire pour estimer le fond de ciel et
    un point légèrement au-dessus, tout en essayant d'éviter les étoiles brillantes.

    Args:
        image_data_2d_float32 (np.ndarray): Image 2D (un canal), dtype float32.
        low_percentile (float): Percentile inférieur (ex: 25.0 pour le fond de ciel).
        high_percentile (float): Percentile supérieur (ex: 90.0 pour un point au-dessus du fond).
        progress_callback (callable, optional): Fonction de callback pour les logs.

    Returns:
        tuple[float, float]: (stat_low, stat_high). Retourne (0.0, 1.0) en cas d'erreur majeure.
    """
    # Define a local alias for the callback for brevity and safety
    # Uses _internal_logger as a fallback if progress_callback is None
    _pcb = lambda msg_key, lvl="DEBUG_VERY_DETAIL", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else _internal_logger.debug(f"PCB_FALLBACK_{lvl}: {msg_key} {kwargs}")

    if not isinstance(image_data_2d_float32, np.ndarray) or image_data_2d_float32.ndim != 2:
        _pcb("stathelper_error_invalid_input_for_stats", lvl="WARN",
             shape=image_data_2d_float32.shape if hasattr(image_data_2d_float32, 'shape') else 'N/A',
             ndim=image_data_2d_float32.ndim if hasattr(image_data_2d_float32, 'ndim') else 'N/A')
        return 0.0, 1.0 # Fallback pour une entrée clairement incorrecte

    if image_data_2d_float32.size == 0:
        _pcb("stathelper_error_empty_image_for_stats", lvl="WARN")
        return 0.0, 1.0

    # Assurer que les données sont finies pour le calcul des percentiles
    # np.nanpercentile gère déjà les NaNs, mais il est bon de savoir si tout est non-fini.
    finite_data = image_data_2d_float32[np.isfinite(image_data_2d_float32)]
    if finite_data.size == 0:
        _pcb("stathelper_warn_all_nan_or_inf_for_stats", lvl="WARN")
        return 0.0, 1.0 # Pas de données valides pour calculer les percentiles

    try:
        # np.nanpercentile est robuste aux NaNs
        stat_low = float(np.nanpercentile(image_data_2d_float32, low_percentile))
        stat_high = float(np.nanpercentile(image_data_2d_float32, high_percentile))
        # _pcb(f"stathelper_debug_percentiles_calculated: low_p={low_percentile}% -> {stat_low:.3g}, high_p={high_percentile}% -> {stat_high:.3g}", lvl="DEBUG_VERY_DETAIL")

    except Exception as e_perc:
        _pcb(f"stathelper_error_percentile_calc: {e_perc}", lvl="WARN")
        # Fallback très simple si nanpercentile échoue pour une raison imprévue
        # (normalement, ne devrait pas arriver si finite_data n'est pas vide)
        # Utilise les min/max des données finies comme un pis-aller.
        if finite_data.size > 0 : # Double check, bien que déjà fait avant
             stat_low = float(np.min(finite_data))
             stat_high = float(np.max(finite_data))
             _pcb(f"stathelper_warn_percentile_exception_fallback_minmax: low={stat_low:.3g}, high={stat_high:.3g}", lvl="WARN")
        else: # Ne devrait jamais être atteint si la logique précédente est correcte
            return 0.0, 1.0


    # Gérer le cas où l'image est (presque) plate
    if abs(stat_high - stat_low) < 1e-5: # 1e-5 est un seuil arbitraire, pourrait être ajusté
        _pcb(f"stathelper_warn_stats_nearly_equal: low={stat_low:.3g}, high={stat_high:.3g}. L'image est peut-être plate ou a peu de dynamique dans les percentiles choisis.", lvl="DEBUG_DETAIL")
        # Si les stats sont égales, cela signifie que la plage entre low_percentile et high_percentile est très étroite.
        # Cela peut arriver avec des images avec peu de signal ou très bruitées où les percentiles tombent au même endroit.
        # On retourne quand même ces valeurs, la logique appelante devra gérer cela (ex: a = 1, b = offset).

    return stat_low, stat_high



def align_images_in_group(image_data_list: list,
                          reference_image_index: int = 0,
                          detection_sigma: float = 3.0,
                          min_area: int = 5,
                          propagate_mask: bool = False, 
                          progress_callback: callable = None) -> list:
    """
    Aligne une liste d'images (données NumPy HWC, float32, ADU) sur une image de référence
    de ce même groupe en utilisant astroalign.
    """
    # Define a local alias for the callback
    _pcb = lambda msg_key, lvl="INFO_DETAIL", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else _internal_logger.debug(f"PCB_FALLBACK_{lvl}: {msg_key} {kwargs}")

    if not ASTROALIGN_AVAILABLE or astroalign_module is None:
        _pcb("aligngroup_error_astroalign_unavailable", lvl="ERROR")
        return [None] * len(image_data_list)

    if not image_data_list or not (0 <= reference_image_index < len(image_data_list)):
        _pcb("aligngroup_error_invalid_input_list_or_ref_index", lvl="ERROR", ref_idx=reference_image_index)
        return [None] * len(image_data_list)

    reference_image_adu = image_data_list[reference_image_index]
    if reference_image_adu is None:
        _pcb("aligngroup_error_ref_image_none", lvl="ERROR", ref_idx=reference_image_index)
        return [None] * len(image_data_list)
    
    if reference_image_adu.dtype != np.float32:
        _pcb(f"AlignGroup: Image de référence (index {reference_image_index}) convertie en float32.", lvl="DEBUG_DETAIL")
        reference_image_adu = reference_image_adu.astype(np.float32)

    _pcb(f"AlignGroup: Alignement intra-tuile sur réf. idx {reference_image_index} (shape {reference_image_adu.shape}).", lvl="DEBUG")
    aligned_images = [None] * len(image_data_list)

    for i, source_image_adu_orig in enumerate(image_data_list):
        if source_image_adu_orig is None:
            _pcb(f"AlignGroup: Image source {i} est None, ignorée.", lvl="WARN")
            continue

        source_image_adu = source_image_adu_orig.astype(np.float32, copy=False) 

        if i == reference_image_index:
            aligned_images[i] = reference_image_adu.copy() 
            _pcb(f"AlignGroup: Image {i} est la référence, copiée.", lvl="DEBUG_DETAIL")
            continue

        _pcb(f"AlignGroup: Alignement image {i} (shape {source_image_adu.shape}) sur référence...", lvl="DEBUG_DETAIL")
        try:
            aligned_image_output, footprint_mask = astroalign_module.register(
                source=source_image_adu, target=reference_image_adu,
                detection_sigma=detection_sigma, min_area=min_area,
                propagate_mask=propagate_mask
            )
            if aligned_image_output is not None:
                if aligned_image_output.shape != reference_image_adu.shape:
                    _pcb("aligngroup_warn_shape_mismatch_after_align", lvl="WARN", img_idx=i, 
                              aligned_shape=aligned_image_output.shape, ref_shape=reference_image_adu.shape)
                    aligned_images[i] = None
                else:
                    aligned_images[i] = aligned_image_output.astype(np.float32)
                    _pcb(f"AlignGroup: Image {i} alignée.", lvl="DEBUG_DETAIL")
            else:
                _pcb("aligngroup_warn_register_returned_none", lvl="WARN", img_idx=i)
                aligned_images[i] = None
        except astroalign_module.MaxIterError:
            _pcb("aligngroup_warn_max_iter_error", lvl="WARN", img_idx=i)
            aligned_images[i] = None
        except ValueError as ve:
            _pcb("aligngroup_warn_value_error", lvl="WARN", img_idx=i, error=str(ve))
            aligned_images[i] = None
        except Exception as e_align:
            _pcb("aligngroup_error_exception_aligning", lvl="ERROR", img_idx=i, error_type=type(e_align).__name__, error_msg=str(e_align))
            _pcb(f"AlignGroup Traceback: {traceback.format_exc()}", lvl="DEBUG_DETAIL")
            aligned_images[i] = None
    return aligned_images



def _normalize_images_linear_fit(image_list_hwc_float32: list[np.ndarray],
                                 reference_index: int = 0,
                                 low_percentile: float = 25.0,
                                 high_percentile: float = 90.0,
                                 progress_callback: callable = None):
    _pcb = lambda msg_key, lvl="INFO_DETAIL", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else _internal_logger.debug(f"PCB_FALLBACK_{lvl}: {msg_key} {kwargs}")
    _pcb("norm_linear_fit_starting", lvl="DEBUG", num_images=len(image_list_hwc_float32), ref_idx=reference_index, low_p=low_percentile, high_p=high_percentile)
    if not image_list_hwc_float32:
        _pcb("norm_linear_fit_error_no_images", lvl="WARN"); return []
    if not (0 <= reference_index < len(image_list_hwc_float32) and image_list_hwc_float32[reference_index] is not None):
        _pcb("norm_linear_fit_error_invalid_ref_index_or_ref_none", lvl="ERROR", ref_idx=reference_index)
        return [img.copy() if img is not None else None for img in image_list_hwc_float32]

    ref_image_hwc_float32 = image_list_hwc_float32[reference_index]
    if ref_image_hwc_float32.dtype != np.float32:
        ref_image_hwc_float32 = ref_image_hwc_float32.astype(np.float32, copy=False)
    is_color = ref_image_hwc_float32.ndim == 3 and ref_image_hwc_float32.shape[-1] == 3
    num_channels = 3 if is_color else 1
    normalized_image_list = [None] * len(image_list_hwc_float32)
    ref_stats_per_channel = []
    for c_idx_ref in range(num_channels):
        ref_channel_2d = ref_image_hwc_float32[..., c_idx_ref] if is_color else ref_image_hwc_float32
        ref_low, ref_high = _calculate_robust_stats_for_linear_fit(ref_channel_2d, low_percentile, high_percentile, progress_callback)
        ref_stats_per_channel.append((ref_low, ref_high))
        _pcb(f"NormLinFit: Réf. Canal {c_idx_ref}: StatLow={ref_low:.3g}, StatHigh={ref_high:.3g}", lvl="DEBUG_DETAIL")
        if abs(ref_high - ref_low) < 1e-5:
             _pcb(f"NormLinFit: AVERT Réf. Canal {c_idx_ref} est (presque) plat.", lvl="WARN")

    for i, src_image_hwc_orig_float32 in enumerate(image_list_hwc_float32):
        if src_image_hwc_orig_float32 is None: continue
        if i == reference_index:
            normalized_image_list[i] = ref_image_hwc_float32.copy()
            continue
        src_image_hwc_float32 = src_image_hwc_orig_float32
        if src_image_hwc_float32.dtype != np.float32:
            src_image_hwc_float32 = src_image_hwc_float32.astype(np.float32, copy=True)
        else:
            src_image_hwc_float32 = src_image_hwc_float32.copy()
        if src_image_hwc_float32.shape != ref_image_hwc_float32.shape:
            _pcb(f"NormLinFit: AVERT Img {i} shape {src_image_hwc_float32.shape} != réf {ref_image_hwc_float32.shape}. Ignorée.", lvl="WARN")
            continue
        for c_idx_src in range(num_channels):
            src_channel_2d = src_image_hwc_float32[..., c_idx_src] if is_color else src_image_hwc_float32
            ref_low, ref_high = ref_stats_per_channel[c_idx_src]
            src_low, src_high = _calculate_robust_stats_for_linear_fit(src_channel_2d, low_percentile, high_percentile, progress_callback)
            a = 1.0; b = 0.0
            delta_src = src_high - src_low; delta_ref = ref_high - ref_low
            if abs(delta_src) > 1e-5:
                if abs(delta_ref) > 1e-5: a = delta_ref / delta_src; b = ref_low - a * src_low
                else: b = ref_low - src_low # a=1
            else:
                if abs(delta_ref) > 1e-5: a = 0.0; b = ref_low
                else: b = ref_low - src_low # a=1
            if abs(a - 1.0) > 1e-3 or abs(b) > 1e-3 * max(abs(ref_low), abs(src_low), 1.0):
                 _pcb(f"NormLinFit: Img {i} C{c_idx_src}: Src(L/H)=({src_low:.3g}/{src_high:.3g}) -> Coeffs a={a:.3f}, b={b:.3f}", lvl="DEBUG_DETAIL")
            transformed_channel = a * src_channel_2d + b
            if is_color: src_image_hwc_float32[..., c_idx_src] = transformed_channel
            else: src_image_hwc_float32 = transformed_channel
        normalized_image_list[i] = src_image_hwc_float32
    _pcb("norm_linear_fit_finished", lvl="DEBUG", num_normalized_successfully=sum(1 for img in normalized_image_list if img is not None))
    return normalized_image_list



# Dans zemosaic_align_stack.py

def _normalize_images_sky_mean(image_list: list[np.ndarray | None], 
                               reference_index: int = 0,
                               sky_percentile: float = 25.0, # Percentile pour estimer le fond de ciel
                               progress_callback: callable = None) -> list[np.ndarray | None]:
    """
    Normalise une liste d'images en ajustant leur fond de ciel moyen (estimé par percentile)
    pour correspondre à celui de l'image de référence.
    Opère sur la luminance pour les images couleur.
    """
    _pcb = lambda msg_key, lvl="INFO_DETAIL", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else _internal_logger.debug(f"PCB_FALLBACK_{lvl}: {msg_key} {kwargs}")

    if not image_list:
        _pcb("norm_skymean_error_no_images", lvl="WARN")
        return []
    
    if not (0 <= reference_index < len(image_list) and image_list[reference_index] is not None):
        _pcb("norm_skymean_error_invalid_ref", lvl="ERROR", ref_idx=reference_index)
        # Retourner une copie des images originales si la référence est invalide
        return [img.copy() if img is not None else None for img in image_list]

    _pcb(f"NormSkyMean: Début normalisation par fond de ciel (percentile {sky_percentile}%) sur réf. idx {reference_index}.", lvl="DEBUG")
    
    ref_image_adu = image_list[reference_index]
    # S'assurer que l'image de référence est en float32 pour les calculs
    if ref_image_adu.dtype != np.float32:
        ref_image_adu_float = ref_image_adu.astype(np.float32, copy=True)
    else:
        ref_image_adu_float = ref_image_adu # Pas besoin de copier si déjà float32 et on ne le modifie pas directement

    # --- Calculer le fond de ciel de référence ---
    ref_sky_level = None
    target_data_for_ref_sky = None
    if ref_image_adu_float.ndim == 3 and ref_image_adu_float.shape[-1] == 3: # Couleur HWC
        luminance_ref = 0.299 * ref_image_adu_float[..., 0] + \
                        0.587 * ref_image_adu_float[..., 1] + \
                        0.114 * ref_image_adu_float[..., 2]
        target_data_for_ref_sky = luminance_ref
    elif ref_image_adu_float.ndim == 2: # Monochrome HW
        target_data_for_ref_sky = ref_image_adu_float
    
    if target_data_for_ref_sky is not None and target_data_for_ref_sky.size > 0:
        try:
            # Utiliser nanpercentile pour être robuste aux NaNs potentiels
            ref_sky_level = np.nanpercentile(target_data_for_ref_sky, sky_percentile)
            _pcb(f"NormSkyMean: Fond de ciel de référence (img idx {reference_index}) estimé à {ref_sky_level:.3g}", lvl="DEBUG_DETAIL")
        except Exception as e_perc_ref:
            _pcb(f"NormSkyMean: Erreur calcul percentile réf: {e_perc_ref}", lvl="WARN")
            # Si échec, on ne peut pas normaliser, retourner les images telles quelles (ou des copies)
            return [img.copy() if img is not None else None for img in image_list]
    else:
        _pcb("NormSkyMean: Impossible de déterminer les données pour le fond de ciel de référence.", lvl="WARN")
        return [img.copy() if img is not None else None for img in image_list]

    if ref_sky_level is None or not np.isfinite(ref_sky_level):
        _pcb(f"NormSkyMean: Fond de ciel de référence invalide ({ref_sky_level}). Normalisation annulée.", lvl="ERROR")
        return [img.copy() if img is not None else None for img in image_list]

    # --- Normaliser chaque image ---
    normalized_image_list = [None] * len(image_list)
    for i, current_image_adu in enumerate(image_list):
        if current_image_adu is None:
            normalized_image_list[i] = None
            continue

        # Faire une copie pour la modification, s'assurer qu'elle est float32
        if current_image_adu.dtype != np.float32:
            img_to_normalize_float = current_image_adu.astype(np.float32, copy=True)
        else:
            img_to_normalize_float = current_image_adu.copy() # Toujours copier pour modifier

        if i == reference_index:
            normalized_image_list[i] = img_to_normalize_float # C'est déjà la référence (ou sa copie float32)
            _pcb(f"NormSkyMean: Image {i} est la référence, copiée.", lvl="DEBUG_VERY_DETAIL")
            continue

        target_data_for_current_sky = None
        is_current_color = img_to_normalize_float.ndim == 3 and img_to_normalize_float.shape[-1] == 3
        if is_current_color:
            luminance_current = 0.299 * img_to_normalize_float[..., 0] + \
                                0.587 * img_to_normalize_float[..., 1] + \
                                0.114 * img_to_normalize_float[..., 2]
            target_data_for_current_sky = luminance_current
        elif img_to_normalize_float.ndim == 2:
            target_data_for_current_sky = img_to_normalize_float
        
        current_sky_level = None
        if target_data_for_current_sky is not None and target_data_for_current_sky.size > 0 :
            try:
                current_sky_level = np.nanpercentile(target_data_for_current_sky, sky_percentile)
            except Exception as e_perc_curr:
                 _pcb(f"NormSkyMean: Erreur calcul percentile image {i}: {e_perc_curr}. Image non normalisée.", lvl="WARN")
                 normalized_image_list[i] = img_to_normalize_float # Retourner la copie non modifiée
                 continue
        
        if current_sky_level is not None and np.isfinite(current_sky_level):
            offset = ref_sky_level - current_sky_level
            img_to_normalize_float += offset # Appliquer l'offset à tous les canaux si couleur, ou à l'image si mono
            normalized_image_list[i] = img_to_normalize_float
            _pcb(f"NormSkyMean: Image {i}, fond_ciel={current_sky_level:.3g}, offset_appliqué={offset:.3g}", lvl="DEBUG_VERY_DETAIL")
        else:
            _pcb(f"NormSkyMean: Fond de ciel invalide pour image {i} ({current_sky_level}). Image non normalisée.", lvl="WARN")
            normalized_image_list[i] = img_to_normalize_float # Retourner la copie non modifiée

    _pcb("NormSkyMean: Normalisation par fond de ciel terminée.", lvl="DEBUG")
    return normalized_image_list



def _calculate_image_weights_noise_variance(image_list: list[np.ndarray | None], 
                                            progress_callback: callable = None) -> list[np.ndarray | None]:
    """
    Calcule les poids pour une liste d'images basés sur l'inverse de la variance du bruit.
    Le bruit est estimé à partir des statistiques sigma-clippées.
    Pour les images couleur, les poids sont calculés et appliqués PAR CANAL.
    Retourne une liste de tableaux de poids (HWC ou HW), de même forme que les images d'entrée.
    """
    _pcb = lambda msg_key, lvl="INFO_DETAIL", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else _internal_logger.debug(f"PCB_FALLBACK_{lvl}: {msg_key} {kwargs}")

    if not image_list:
        _pcb("weight_noisevar_error_no_images", lvl="WARN")
        return []

    if not SIGMA_CLIP_AVAILABLE or sigma_clipped_stats_func is None:
        _pcb("weight_noisevar_warn_astropy_stats_unavailable", lvl="WARN")
        weights = []
        for img in image_list:
            if img is not None:
                weights.append(np.ones_like(img, dtype=np.float32))
            else:
                weights.append(None)
        return weights

    # Va stocker pour chaque image valide:
    # - Pour les images couleur: une liste [var_R, var_G, var_B]
    # - Pour les images monochrome: une liste [var_Mono]
    variances_per_image_channels = [] 
    valid_image_indices = []

    _pcb(f"WeightNoiseVar: Début calcul des poids (par canal si couleur) pour {len(image_list)} images.", lvl="DEBUG")

    for i, image_data_adu in enumerate(image_list):
        if image_data_adu is None:
            continue # Sera géré à la fin
        
        if image_data_adu.size == 0:
            _pcb(f"WeightNoiseVar: Image {i} est vide, poids non calculé pour celle-ci.", lvl="WARN")
            continue

        img_for_stats = image_data_adu
        if image_data_adu.dtype != np.float32:
            # Il est crucial de faire une copie si on change le type pour ne pas modifier l'original dans image_list
            img_for_stats = image_data_adu.astype(np.float32, copy=True) 

        current_image_channel_variances = []
        num_channels_in_image = 0

        if img_for_stats.ndim == 3 and img_for_stats.shape[-1] == 3: # Image couleur HWC
            num_channels_in_image = img_for_stats.shape[-1]
            for c_idx in range(num_channels_in_image):
                channel_data = img_for_stats[..., c_idx]
                if channel_data.size == 0:
                    _pcb(f"WeightNoiseVar: Image {i}, Canal {c_idx} vide.", lvl="WARN")
                    current_image_channel_variances.append(np.inf) # Poids sera quasi nul
                    continue
                try:
                    # Utiliser sigma_lower, sigma_upper pour un écrêtage plus robuste
                    _, _, stddev_ch = sigma_clipped_stats_func(
                        channel_data, sigma_lower=3.0, sigma_upper=3.0, maxiters=5
                    )
                    if stddev_ch is not None and np.isfinite(stddev_ch) and stddev_ch > 1e-9: # 1e-9 pour éviter variance nulle
                        current_image_channel_variances.append(stddev_ch**2)
                    else:
                        _pcb(f"WeightNoiseVar: Image {i}, Canal {c_idx}, stddev invalide ({stddev_ch}). Variance Inf.", lvl="WARN")
                        current_image_channel_variances.append(np.inf)
                except Exception as e_stats_ch:
                    _pcb(f"WeightNoiseVar: Erreur stats image {i}, canal {c_idx}: {e_stats_ch}", lvl="WARN")
                    current_image_channel_variances.append(np.inf)
            
        elif img_for_stats.ndim == 2: # Image monochrome HW
            num_channels_in_image = 1 # Conceptuellement
            if img_for_stats.size == 0:
                _pcb(f"WeightNoiseVar: Image monochrome {i} vide.", lvl="WARN")
                current_image_channel_variances.append(np.inf)
            else:
                try:
                    _, _, stddev = sigma_clipped_stats_func(
                        img_for_stats, sigma_lower=3.0, sigma_upper=3.0, maxiters=5
                    )
                    if stddev is not None and np.isfinite(stddev) and stddev > 1e-9:
                        current_image_channel_variances.append(stddev**2)
                    else:
                        _pcb(f"WeightNoiseVar: Image monochrome {i}, stddev invalide ({stddev}). Variance Inf.", lvl="WARN")
                        current_image_channel_variances.append(np.inf)
                except Exception as e_stats_mono:
                    _pcb(f"WeightNoiseVar: Erreur stats image monochrome {i}: {e_stats_mono}", lvl="WARN")
                    current_image_channel_variances.append(np.inf)
        else:
            _pcb(f"WeightNoiseVar: Image {i} a une forme non supportée ({img_for_stats.shape}).", lvl="WARN")
            continue # Passe à l'image suivante
        
        # Si on a réussi à calculer des variances pour les canaux de cette image
        if len(current_image_channel_variances) == num_channels_in_image and num_channels_in_image > 0:
            variances_per_image_channels.append(current_image_channel_variances)
            valid_image_indices.append(i)
        elif num_channels_in_image > 0 : # Si on s'attendait à des canaux mais on n'a pas toutes les variances
             _pcb(f"WeightNoiseVar: N'a pas pu calculer toutes les variances de canal pour l'image {i}.", lvl="WARN")


    if not variances_per_image_channels:
        _pcb("weight_noisevar_warn_no_variances_calculated_at_all", lvl="WARN")
        return [np.ones_like(img, dtype=np.float32) if img is not None else None for img in image_list]

    all_finite_variances = []
    for var_list_for_img in variances_per_image_channels:
        for var_val in var_list_for_img:
            if np.isfinite(var_val) and var_val > 1e-18: # Seuil très bas pour variance valide
                all_finite_variances.append(var_val)
    
    min_overall_variance = np.min(all_finite_variances) if all_finite_variances else 1e-9
    if min_overall_variance <= 0: min_overall_variance = 1e-9 # Assurer qu'elle est positive

    _pcb(f"WeightNoiseVar: Variance minimale globale trouvée: {min_overall_variance:.3g}", lvl="DEBUG_DETAIL")

    output_weights_list = [None] * len(image_list)

    for idx_in_valid_arrays, original_image_idx in enumerate(valid_image_indices):
        original_img_data_shape_ref = image_list[original_image_idx] # Pour obtenir la shape HWC ou HW
        if original_img_data_shape_ref is None: continue # Ne devrait pas arriver

        variances_for_current_img = variances_per_image_channels[idx_in_valid_arrays]
        
        # Créer le tableau de poids pour cette image, de la même forme
        weights_for_this_img_array = np.zeros_like(original_img_data_shape_ref, dtype=np.float32)

        if original_img_data_shape_ref.ndim == 3 and len(variances_for_current_img) == original_img_data_shape_ref.shape[-1]: # Couleur
            for c_idx in range(original_img_data_shape_ref.shape[-1]):
                variance_ch = variances_for_current_img[c_idx]
                if np.isfinite(variance_ch) and variance_ch > 1e-18:
                    calculated_weight = min_overall_variance / variance_ch
                else:
                    calculated_weight = 1e-6 # Poids très faible si variance du canal est invalide ou nulle
                weights_for_this_img_array[..., c_idx] = calculated_weight
                # _pcb(f"WeightNoiseVar: Img {original_image_idx}, Ch {c_idx}, Var={variance_ch:.2e}, PoidsRel={calculated_weight:.3f}", lvl="DEBUG_VERY_DETAIL")
        
        elif original_img_data_shape_ref.ndim == 2 and len(variances_for_current_img) == 1: # Monochrome
            variance_mono = variances_for_current_img[0]
            if np.isfinite(variance_mono) and variance_mono > 1e-18:
                calculated_weight = min_overall_variance / variance_mono
            else:
                calculated_weight = 1e-6
            weights_for_this_img_array[:] = calculated_weight # Appliquer à tous les pixels de l'image HW
            # _pcb(f"WeightNoiseVar: Img {original_image_idx} (Mono), Var={variance_mono:.2e}, PoidsRel={calculated_weight:.3f}", lvl="DEBUG_VERY_DETAIL")
        
        output_weights_list[original_image_idx] = weights_for_this_img_array

    # Pour les images qui n'ont pas pu être traitées (initialement None, ou erreur en cours de route)
    for i in range(len(image_list)):
        if output_weights_list[i] is None and image_list[i] is not None:
            _pcb(f"WeightNoiseVar: Image {i} (pas de poids valide calc.), fallback sur poids uniforme 1.0.", lvl="DEBUG_DETAIL")
            output_weights_list[i] = np.ones_like(image_list[i], dtype=np.float32)
            
    num_actual_weights = sum(1 for w_arr in output_weights_list if w_arr is not None)
    _pcb(f"WeightNoiseVar: Calcul des poids (par canal si couleur) terminé. {num_actual_weights}/{len(image_list)} tableaux de poids retournés.", lvl="DEBUG")
    return output_weights_list



def _estimate_initial_fwhm(data_2d: np.ndarray, progress_callback: callable = None) -> float:
    """
    Tente d'estimer une FWHM initiale à partir des données 2D.
    Utilise la segmentation et les propriétés des sources.
    """
    _pcb_est = lambda msg_key, lvl="INFO_DETAIL", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else _internal_logger.debug(f"PCB_FALLBACK_{lvl}: {msg_key} {kwargs}")

    default_fwhm = 4.0 # Valeur de secours
    if data_2d.size < 1000: # Pas assez de données pour une estimation fiable
        _pcb_est("fwhm_est_data_insufficient", lvl="DEBUG_DETAIL", returned_fwhm=default_fwhm)
        return default_fwhm

    try:
        # Estimation simple du fond et du bruit pour la segmentation
        _, median, std = sigma_clipped_stats_func(data_2d, sigma=3.0, maxiters=5)
        if not (np.isfinite(median) and np.isfinite(std) and std > 1e-6):
            _pcb_est("fwhm_est_stats_invalid", lvl="DEBUG_DETAIL", returned_fwhm=default_fwhm)
            return default_fwhm
            
        threshold_seg_est = median + (3.0 * std) # Seuil pour la segmentation
        
        # Segmentation
        segm_map = detect_sources(data_2d, threshold_seg_est, npixels=5) # npixels minimum pour une source
        if segm_map is None:
            _pcb_est("fwhm_est_segmentation_failed", lvl="DEBUG_DETAIL", returned_fwhm=default_fwhm)
            return default_fwhm

        # Catalogue des sources
        cat = SourceCatalog(data_2d, segm_map)
        if not cat or len(cat) == 0:
            _pcb_est("fwhm_est_no_sources_in_catalog", lvl="DEBUG_DETAIL", returned_fwhm=default_fwhm)
            return default_fwhm

        fwhms_from_cat = []
        # On ne prend que les sources "raisonnables" pour l'estimation
        for props in cat:
            try:
                # equivalent_fwhm est une bonne estimation si la source est ~gaussienne
                # On filtre sur l'ellipticité pour ne garder que les sources rondes
                if props.eccentricity is not None and props.eccentricity < 0.5 and \
                   props.equivalent_fwhm is not None and np.isfinite(props.equivalent_fwhm) and \
                   1.0 < props.equivalent_fwhm < 20.0: # FWHM doit être dans une plage plausible
                    fwhms_from_cat.append(props.equivalent_fwhm.value)
            except AttributeError: # Certaines propriétés peuvent manquer
                continue
            if len(fwhms_from_cat) >= 100: # Limiter le nombre de sources pour l'estimation
                break
        
        if not fwhms_from_cat:
            _pcb_est("fwhm_est_no_valid_fwhm_from_cat", lvl="DEBUG_DETAIL", returned_fwhm=default_fwhm)
            return default_fwhm
            
        estimated_fwhm = np.nanmedian(fwhms_from_cat)
        if np.isfinite(estimated_fwhm) and 1.0 < estimated_fwhm < 15.0:
            _pcb_est("fwhm_est_success", lvl="DEBUG_DETAIL", estimated_fwhm=float(estimated_fwhm))
            return float(estimated_fwhm)
        else:
            _pcb_est("fwhm_est_median_invalid", lvl="DEBUG_DETAIL", median_fwhm=estimated_fwhm, returned_fwhm=default_fwhm)
            return default_fwhm

    except Exception as e_est:
        _pcb_est("fwhm_est_exception", lvl="WARN", error=str(e_est), returned_fwhm=default_fwhm)
        return default_fwhm


def _calculate_image_weights_noise_fwhm(image_list: list[np.ndarray | None], 
                                        progress_callback: callable = None) -> list[np.ndarray | None]:
    """
    Calcule les poids pour une liste d'images basés sur l'inverse de la FWHM moyenne des étoiles.
    Tente d'estimer une FWHM initiale pour la détection de sources.
    """
    _pcb = lambda msg_key, lvl="INFO_DETAIL", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else _internal_logger.debug(f"PCB_FALLBACK_{lvl}: {msg_key} {kwargs}")

    if not image_list:
        _pcb("weight_fwhm_error_no_images", lvl="WARN")
        return []

    if not PHOTOUTILS_AVAILABLE or not SIGMA_CLIP_AVAILABLE:
        missing_fwhm_deps = []
        if not PHOTOUTILS_AVAILABLE: missing_fwhm_deps.append("Photutils")
        if not SIGMA_CLIP_AVAILABLE: missing_fwhm_deps.append("Astropy.stats (SigmaClip)")
        _pcb("weight_fwhm_warn_deps_unavailable", lvl="WARN", missing=", ".join(missing_fwhm_deps))
        return [np.ones_like(img, dtype=np.float32) if img is not None else None for img in image_list]

    fwhm_values_per_image = [] 
    valid_image_indices_fwhm = []

    _pcb(f"WeightFWHM: Début calcul des poids FWHM pour {len(image_list)} images.", lvl="DEBUG")

    for i, image_data_adu in enumerate(image_list):
        if image_data_adu is None:
            continue
        
        if image_data_adu.size == 0:
            _pcb("weight_fwhm_img_empty", lvl="WARN", img_idx=i)
            continue

        img_for_fwhm_calc = image_data_adu
        if image_data_adu.dtype != np.float32:
            img_for_fwhm_calc = image_data_adu.astype(np.float32, copy=True)

        target_data_for_fwhm = None
        if img_for_fwhm_calc.ndim == 3 and img_for_fwhm_calc.shape[-1] == 3:
            luminance = 0.299 * img_for_fwhm_calc[..., 0] + \
                        0.587 * img_for_fwhm_calc[..., 1] + \
                        0.114 * img_for_fwhm_calc[..., 2]
            target_data_for_fwhm = luminance
        elif img_for_fwhm_calc.ndim == 2:
            target_data_for_fwhm = img_for_fwhm_calc
        else:
            _pcb("weight_fwhm_unsupported_shape", lvl="WARN", img_idx=i, shape=img_for_fwhm_calc.shape)
            continue
        
        if target_data_for_fwhm is None or target_data_for_fwhm.size < 50*50: # Besoin d'une taille minimale
             _pcb("weight_fwhm_insufficient_data", lvl="WARN", img_idx=i)
             continue
        
        try:
            estimated_initial_fwhm = _estimate_initial_fwhm(target_data_for_fwhm, progress_callback)
            _pcb(f"WeightFWHM: Image {i}, FWHM initiale estimée pour détection: {estimated_initial_fwhm:.2f} px", lvl="DEBUG_DETAIL")

            box_size_bg = min(target_data_for_fwhm.shape[0] // 8, target_data_for_fwhm.shape[1] // 8, 50)
            box_size_bg = max(box_size_bg, 16)
            
            sigma_clip_bg_obj = SigmaClip(sigma=3.0) # Renommé pour éviter conflit
            bkg_estimator_obj = MedianBackground()   # Renommé pour éviter conflit
            
            if not np.any(np.isfinite(target_data_for_fwhm)):
                _pcb("weight_fwhm_no_finite_data", lvl="WARN", img_idx=i)
                fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i); continue
            
            std_data_check = np.nanstd(target_data_for_fwhm)
            if std_data_check < 1e-6 :
                 _pcb("weight_fwhm_image_flat", lvl="DEBUG_DETAIL", img_idx=i, stddev=std_data_check)
                 fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i); continue

            bkg_obj = None # Pour vérifier si bkg a été défini
            try:
                bkg_obj = Background2D(target_data_for_fwhm, (box_size_bg, box_size_bg), 
                                   filter_size=(3, 3), sigma_clip=sigma_clip_bg_obj, bkg_estimator=bkg_estimator_obj)
                data_subtracted = target_data_for_fwhm - bkg_obj.background
                threshold_daofind_val = 5.0 * bkg_obj.background_rms 
            except (ValueError, TypeError) as ve_bkg: 
                _pcb("weight_fwhm_bkg2d_error", lvl="WARN", img_idx=i, error=str(ve_bkg))
                _, median_glob, stddev_glob = sigma_clipped_stats_func(target_data_for_fwhm, sigma=3.0, maxiters=5)
                if not (np.isfinite(median_glob) and np.isfinite(stddev_glob)):
                    _pcb("weight_fwhm_global_stats_invalid", lvl="WARN", img_idx=i)
                    fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i); continue
                data_subtracted = target_data_for_fwhm - median_glob
                threshold_daofind_val = 5.0 * stddev_glob

            # S'assurer que threshold_daofind_val est un scalaire positif
            if hasattr(threshold_daofind_val, 'mean'): threshold_daofind_val = np.abs(np.mean(threshold_daofind_val))
            else: threshold_daofind_val = np.abs(threshold_daofind_val)
            if threshold_daofind_val < 1e-5 : threshold_daofind_val = 1e-5 # Minimum seuil

            sources_table = None
            try:
                daofind_obj = DAOStarFinder(fwhm=estimated_initial_fwhm, threshold=threshold_daofind_val,
                                        sharplo=0.2, sharphi=1.0, roundlo=-0.8, roundhi=0.8, sky=0.0)
                sources_table = daofind_obj(data_subtracted)
            except Exception as e_daofind:
                 _pcb("weight_fwhm_daofind_error", lvl="WARN", img_idx=i, error=str(e_daofind))
                 fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i); continue

            if sources_table is None or len(sources_table) < 5:
                _pcb("weight_fwhm_not_enough_sources_daofind", lvl="DEBUG_DETAIL", img_idx=i, count=len(sources_table) if sources_table is not None else 0)
                fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i); continue

            # Utilisation de SourceCatalog pour les propriétés morphologiques
            threshold_seg_val = 1.5 * (bkg_obj.background_rms if bkg_obj and hasattr(bkg_obj, 'background_rms') else np.nanstd(data_subtracted))
            if hasattr(threshold_seg_val, 'mean'): threshold_seg_val = np.abs(np.mean(threshold_seg_val))
            else: threshold_seg_val = np.abs(threshold_seg_val)
            if threshold_seg_val < 1e-5 : threshold_seg_val = 1e-5

            segm_map_cat = detect_sources(data_subtracted, threshold_seg_val, npixels=7) # npixels un peu plus grand
            if segm_map_cat is None:
                _pcb("weight_fwhm_segmentation_cat_failed", lvl="DEBUG_DETAIL", img_idx=i)
                fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i); continue
            
            # Filtrer les sources de DAOStarFinder avant de les passer à SourceCatalog
            h_img_cat, w_img_cat = data_subtracted.shape
            border_margin_cat = int(estimated_initial_fwhm * 2) # Marge basée sur FWHM
            
            # Assurer que les colonnes existent avant de filtrer
            cols_to_check = ['xcentroid', 'ycentroid', 'flux', 'sharpness', 'roundness1', 'roundness2']
            if not all(col in sources_table.colnames for col in cols_to_check):
                _pcb("weight_fwhm_missing_daofind_cols", lvl="WARN", img_idx=i, missing_cols=[c for c in cols_to_check if c not in sources_table.colnames])
                fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i); continue


            valid_sources_mask_cat = (
                (sources_table['xcentroid'] > border_margin_cat) &
                (sources_table['xcentroid'] < w_img_cat - border_margin_cat) &
                (sources_table['ycentroid'] > border_margin_cat) &
                (sources_table['ycentroid'] < h_img_cat - border_margin_cat) &
                (sources_table['sharpness'] > 0.3) & (sources_table['sharpness'] < 0.95) & # Sources nettes mais pas trop
                (np.abs(sources_table['roundness1']) < 0.3) & (np.abs(sources_table['roundness2']) < 0.3) # Assez rondes
            )
            filtered_sources_table = sources_table[valid_sources_mask_cat]
            
            if not filtered_sources_table or len(filtered_sources_table) < 3:
                _pcb("weight_fwhm_not_enough_sources_after_filter_dao", lvl="DEBUG_DETAIL", img_idx=i)
                fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i); continue

            # Trier par flux et prendre les N plus brillantes
            filtered_sources_table.sort('flux', reverse=True)
            top_sources_table = filtered_sources_table[:100] # Limiter aux 100 plus brillantes
            
            # Passer les positions des sources détectées par DAOStarFinder à SourceCatalog
            try:
                cat_obj = SourceCatalog(data_subtracted, segm_map_cat, sources=top_sources_table)
            except Exception as e_scat: # SourceCatalog peut échouer si segm_map_cat est incompatible avec sources
                 _pcb("weight_fwhm_sourcecatalog_init_error", lvl="WARN", img_idx=i, error=str(e_scat))
                 fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i); continue


            if not cat_obj or len(cat_obj) == 0:
                _pcb("weight_fwhm_no_sources_in_final_catalog", lvl="DEBUG_DETAIL", img_idx=i)
                fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i); continue

            fwhms_this_image = []
            for source_props in cat_obj:
                try:
                    # equivalent_fwhm est disponible et généralement fiable pour les sources bien segmentées.
                    # On pourrait aussi utiliser (semimajor_axis_sigma + semiminor_axis_sigma) / 2 * gaussian_sigma_to_fwhm
                    fwhm_val = source_props.equivalent_fwhm # C'est déjà une FWHM en pixels
                    if fwhm_val is not None and np.isfinite(fwhm_val) and \
                       0.8 < fwhm_val < (estimated_initial_fwhm * 2.5): # Doit être dans une plage raisonnable
                        fwhms_this_image.append(fwhm_val)
                except AttributeError: continue
                except Exception: continue
            
            if not fwhms_this_image:
                _pcb("weight_fwhm_no_valid_fwhm_from_catalog_props", lvl="DEBUG_DETAIL", img_idx=i)
                fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i); continue

            median_fwhm_val = np.nanmedian(fwhms_this_image)
            if np.isfinite(median_fwhm_val) and 0.7 < median_fwhm_val < 20.0: # FWHM doit être > ~0.7 pixel et < 20
                fwhm_values_per_image.append(median_fwhm_val)
                valid_image_indices_fwhm.append(i)
                _pcb("weight_fwhm_success", lvl="DEBUG_DETAIL", img_idx=i, median_fwhm=median_fwhm_val, num_stars=len(fwhms_this_image))
            else:
                _pcb("weight_fwhm_median_fwhm_invalid", lvl="WARN", img_idx=i, median_fwhm=median_fwhm_val)
                fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i)

        except Exception as e_fwhm_main_loop:
            _pcb("weight_fwhm_mainloop_exception", lvl="ERROR", img_idx=i, error=str(e_fwhm_main_loop))
            _internal_logger.error(f"Traceback FWHM image {i}:", exc_info=True)
            fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i)


    # --- Fin de la boucle sur les images ---

    if not fwhm_values_per_image: # Si aucune FWHM n'a pu être calculée pour aucune image
        _pcb("weight_fwhm_warn_no_fwhm_values_overall", lvl="WARN")
        return [np.ones_like(img, dtype=np.float32) if img is not None else None for img in image_list]

    finite_fwhms_all = [f for f in fwhm_values_per_image if np.isfinite(f) and f > 0.1] # 0.1 seuil très bas
    if not finite_fwhms_all:
        _pcb("weight_fwhm_warn_all_fwhm_are_infinite", lvl="WARN")
        return [np.ones_like(img, dtype=np.float32) if img is not None else None for img in image_list]

    min_overall_valid_fwhm = np.min(finite_fwhms_all)
    if min_overall_valid_fwhm < 0.5 : min_overall_valid_fwhm = 0.5 # FWHM minimale raisonnable

    _pcb(f"WeightFWHM: FWHM minimale globale valide: {min_overall_valid_fwhm:.2f} px", lvl="DEBUG_DETAIL")

    final_calculated_weights_scalar_fwhm = {}
    for idx_in_valid_list, original_idx in enumerate(valid_image_indices_fwhm):
        fwhm_current_image = fwhm_values_per_image[idx_in_valid_list]
        weight_val = 1e-6 # Poids par défaut très faible
        if np.isfinite(fwhm_current_image) and fwhm_current_image > 0.1:
            # Poids = (min_FWHM / FWHM_image) ^ N. Ici N=1.
            # Cela donne un poids de 1 à la meilleure image, <1 aux autres.
            # Si FWHM_image est plus petit que min_overall_valid_fwhm (ne devrait pas arriver), clamp à 1.
            weight_val = min_overall_valid_fwhm / max(fwhm_current_image, min_overall_valid_fwhm)
        final_calculated_weights_scalar_fwhm[original_idx] = weight_val
        _pcb(f"WeightFWHM: Img idx_orig={original_idx}, FWHM={fwhm_current_image:.2f}, PoidsRelFinal={weight_val:.3f}", lvl="DEBUG_DETAIL")

    output_weights_list_fwhm = [None] * len(image_list)
    for i, original_image_data in enumerate(image_list):
        if original_image_data is None:
            output_weights_list_fwhm[i] = None
        elif i in final_calculated_weights_scalar_fwhm:
            scalar_w_fwhm = final_calculated_weights_scalar_fwhm[i]
            output_weights_list_fwhm[i] = np.full_like(original_image_data, scalar_w_fwhm, dtype=np.float32)
        else: 
            _pcb("weight_fwhm_fallback_weight_one", lvl="DEBUG_DETAIL", img_idx=i)
            output_weights_list_fwhm[i] = np.ones_like(original_image_data, dtype=np.float32)
            
    num_actual_weights_fwhm = sum(1 for w_arr in output_weights_list_fwhm if w_arr is not None)
    _pcb(f"WeightFWHM: Calcul des poids FWHM terminé. {num_actual_weights_fwhm}/{len(image_list)} tableaux de poids retournés.", lvl="DEBUG")
    return output_weights_list_fwhm

def _reject_outliers_kappa_sigma(stacked_array_NHDWC, sigma_low, sigma_high, progress_callback=None):
    _pcb = lambda msg_key, lvl="INFO_DETAIL", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else _internal_logger.debug(f"PCB_FALLBACK_{lvl}: {msg_key} {kwargs}")
    
    _pcb(f"RejKappaSigma: Rejet Kappa-Sigma (low={sigma_low}, high={sigma_high}).", lvl="DEBUG")
    if not SIGMA_CLIP_AVAILABLE or sigma_clipped_stats_func is None:
        _pcb("stackhelper_warn_kappa_sigma_astropy_unavailable", lvl="WARN")
        return stacked_array_NHDWC, np.ones_like(stacked_array_NHDWC, dtype=bool) 
    
    rejection_mask = np.ones_like(stacked_array_NHDWC, dtype=bool)
    output_data_with_nans = stacked_array_NHDWC.copy()

    if stacked_array_NHDWC.ndim == 4: # Couleur (N, H, W, C)
        for c in range(stacked_array_NHDWC.shape[3]):
            channel_data = stacked_array_NHDWC[..., c]
            try: 
                _, median_ch, stddev_ch = sigma_clipped_stats_func(channel_data, sigma_lower=sigma_low, sigma_upper=sigma_high, axis=0, maxiters=5)
            except TypeError: 
                _, median_ch, stddev_ch = sigma_clipped_stats_func(channel_data, sigma=max(sigma_low, sigma_high), axis=0, maxiters=5) 
            lower_bound = median_ch - sigma_low * stddev_ch; upper_bound = median_ch + sigma_high * stddev_ch
            channel_rejection_this_iter = (channel_data < lower_bound) | (channel_data > upper_bound)
            rejection_mask[..., c] = ~channel_rejection_this_iter
            output_data_with_nans[channel_rejection_this_iter, c] = np.nan
    elif stacked_array_NHDWC.ndim == 3: # Monochrome (N, H, W)
        try: _, median_img, stddev_img = sigma_clipped_stats_func(stacked_array_NHDWC, sigma_lower=sigma_low, sigma_upper=sigma_high, axis=0, maxiters=5)
        except TypeError: _, median_img, stddev_img = sigma_clipped_stats_func(stacked_array_NHDWC, sigma=max(sigma_low, sigma_high), axis=0, maxiters=5)
        lower_bound = median_img - sigma_low * stddev_img; upper_bound = median_img + sigma_high * stddev_img
        stats_rejection_this_iter = (stacked_array_NHDWC < lower_bound) | (stacked_array_NHDWC > upper_bound)
        rejection_mask = ~stats_rejection_this_iter
        output_data_with_nans[stats_rejection_this_iter] = np.nan
    else:
        _pcb("stackhelper_error_kappa_sigma_unexpected_shape", lvl="ERROR", shape=stacked_array_NHDWC.shape)
        return stacked_array_NHDWC, rejection_mask
    num_rejected = np.sum(~rejection_mask)
    _pcb("stackhelper_info_kappa_sigma_rejected_pixels", lvl="INFO_DETAIL", num_rejected=num_rejected)
    return output_data_with_nans, rejection_mask


def _apply_winsor_single(args):
    """Helper for parallel winsorization.

    Ensures winsorization is applied along the image axis (axis=0) to
    preserve per-pixel statistics. ``scipy.stats.mstats.winsorize`` flattens
    the array when no axis is provided which would incorrectly clip signal
    across the entire stack.
    """
    arr, limits = args
    return np.asarray(winsorize_func(arr, limits, axis=0))


def parallel_rejwinsor(channels, limits, max_workers, progress_callback=None):
    """Apply winsorization in parallel on a list of arrays."""
    args_list = [(ch, limits) for ch in channels]

    if max_workers <= 1 or len(args_list) <= 1:
        results = []
        for idx, a in enumerate(args_list, start=1):
            results.append(_apply_winsor_single(a))
            if progress_callback:
                progress_callback(idx, len(args_list))
        return results

    results = [None] * len(args_list)

    # Avoid spawning a new process pool when already running inside a
    # multiprocessing worker as this would raise "daemonic processes are not
    # allowed to have children". In that case fallback to threads.
    parent_is_daemon = multiprocessing.current_process().daemon
    Executor = ThreadPoolExecutor if parent_is_daemon else ProcessPoolExecutor

    with Executor(max_workers=max_workers) as exe:
        futures = {exe.submit(_apply_winsor_single, a): i for i, a in enumerate(args_list)}
        total = len(futures)
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()
            done += 1
            if progress_callback:
                progress_callback(done, total)

    return results



def _reject_outliers_winsorized_sigma_clip(
    stacked_array_NHDWC: np.ndarray,
    winsor_limits_tuple: tuple[float, float], # (low_cut_fraction, high_cut_fraction), ex: (0.05, 0.05)
    sigma_low: float,
    sigma_high: float,
    progress_callback: callable = None,
    max_workers: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rejette les outliers en utilisant un Winsorized Sigma Clip.
    1. Winsorize les données le long de l'axe des images.
    2. Calcule les statistiques sigma-clippées sur les données winsorisées.
    3. Rejette les pixels des données *originales* basés sur ces statistiques.

    Args:
        stacked_array_NHDWC: Tableau des images empilées (N, H, W, C) ou (N, H, W).
        winsor_limits_tuple: Tuple de fractions (0-0.5) pour écrêter en bas et en haut.
        sigma_low: Nombre de sigmas pour le seuil inférieur de rejet.
        sigma_high: Nombre de sigmas pour le seuil supérieur de rejet.
        progress_callback: Fonction de callback pour les logs.
        max_workers: Nombre maximum de travailleurs parallèles pour la winsorisation.\n            Typiquement issu de ``run_cfg.winsor_worker_limit``.

    Returns:
        tuple[np.ndarray, np.ndarray]: 
            - output_data_with_nans: Les données originales avec NaN où les pixels sont rejetés.
            - rejection_mask: Masque booléen (True où les pixels sont gardés).
    """
    _pcb = lambda msg_key, lvl="INFO_DETAIL", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else _internal_logger.debug(f"PCB_FALLBACK_{lvl}: {msg_key} {kwargs}")

    if not (SCIPY_AVAILABLE and winsorize_func and SIGMA_CLIP_AVAILABLE and sigma_clipped_stats_func):
        missing_deps = []
        if not SCIPY_AVAILABLE or not winsorize_func: missing_deps.append("Scipy (winsorize)")
        if not SIGMA_CLIP_AVAILABLE or not sigma_clipped_stats_func: missing_deps.append("Astropy.stats (sigma_clipped_stats)")
        _pcb("reject_winsor_warn_deps_unavailable", lvl="WARN", missing=", ".join(missing_deps))
        # Retourner les données originales sans rejet si les dépendances manquent
        return stacked_array_NHDWC.copy(), np.ones_like(stacked_array_NHDWC, dtype=bool)

    _pcb(f"RejWinsor: Début Rejet Winsorized Sigma Clip. Limits={winsor_limits_tuple}, SigmaLow={sigma_low}, SigmaHigh={sigma_high}.", lvl="DEBUG",
         shape=stacked_array_NHDWC.shape)

    # S'assurer que les limites de winsorisation sont valides (doit être déjà fait dans la GUI, mais double check)
    low_cut, high_cut = winsor_limits_tuple
    if not (0.0 <= low_cut < 0.5 and 0.0 <= high_cut < 0.5 and (low_cut + high_cut) < 1.0):
        _pcb("reject_winsor_error_invalid_limits", lvl="ERROR", limits=winsor_limits_tuple)
        return stacked_array_NHDWC.copy(), np.ones_like(stacked_array_NHDWC, dtype=bool)

    # Copie des données originales pour y insérer les NaN pour les pixels rejetés
    output_data_with_nans = stacked_array_NHDWC.astype(np.float32, copy=True) # Travailler sur float32
    rejection_mask_final = np.ones_like(stacked_array_NHDWC, dtype=bool) # True = garder

    is_color = stacked_array_NHDWC.ndim == 4 and stacked_array_NHDWC.shape[-1] == 3
    num_images_in_stack = stacked_array_NHDWC.shape[0]

    if num_images_in_stack < 3: # Winsorize et sigma-clip ont besoin d'assez de données
        _pcb("reject_winsor_warn_not_enough_images", lvl="WARN", num_images=num_images_in_stack)
        return output_data_with_nans, rejection_mask_final # Pas de rejet si trop peu d'images

    try:
        if is_color:
            _pcb("RejWinsor: Traitement image couleur (par canal).", lvl="DEBUG_DETAIL")

            orig_channels = [stacked_array_NHDWC[..., idx].astype(np.float32, copy=False)
                             for idx in range(stacked_array_NHDWC.shape[-1])]

            def prog_cb(done, total):
                _pcb("reject_winsor_info_channel_progress", lvl="INFO_DETAIL", channel=done)

            winsorized_channels = parallel_rejwinsor(orig_channels, winsor_limits_tuple,
                                                     max_workers=max_workers, progress_callback=prog_cb)

            for c_idx, winsorized_channel_data in enumerate(winsorized_channels):
                _pcb(f"  RejWinsor: Canal {c_idx}...", lvl="DEBUG_VERY_DETAIL")
                original_channel_data_NHW = orig_channels[c_idx]

                try:
                    _, median_winsorized, stddev_winsorized = sigma_clipped_stats_func(
                        winsorized_channel_data, sigma=3.0, axis=0, maxiters=5
                    )
                except TypeError:
                    _, median_winsorized, stddev_winsorized = sigma_clipped_stats_func(
                        winsorized_channel_data, sigma_lower=3.0, sigma_upper=3.0, axis=0, maxiters=5
                    )

                lower_bound = median_winsorized - (sigma_low * stddev_winsorized)
                upper_bound = median_winsorized + (sigma_high * stddev_winsorized)

                pixels_to_reject_this_channel = (
                    original_channel_data_NHW < lower_bound[np.newaxis, ...]
                ) | (
                    original_channel_data_NHW > upper_bound[np.newaxis, ...]
                )

                rejection_mask_final[..., c_idx] = ~pixels_to_reject_this_channel
                output_data_with_nans[pixels_to_reject_this_channel, c_idx] = np.nan

                num_rejected_ch = np.sum(pixels_to_reject_this_channel)
                _pcb(
                    f"    RejWinsor: Canal {c_idx}, {num_rejected_ch} pixels rejetés.",
                    lvl="DEBUG_DETAIL",
                )
                time.sleep(0)
        else: # Image monochrome (N, H, W)
            _pcb("reject_winsor_info_mono_progress", lvl="INFO_DETAIL")
            _pcb("RejWinsor: Traitement image monochrome.", lvl="DEBUG_DETAIL")
            original_data_NHW = stacked_array_NHDWC.astype(np.float32, copy=False)

            winsorized_data = winsorize_func(original_data_NHW, limits=winsor_limits_tuple, axis=0)
            # _pcb("  Monochrome: Winsorization terminée.", lvl="DEBUG_VERY_DETAIL")
            
            try:
                _, median_winsorized, stddev_winsorized = sigma_clipped_stats_func(
                    winsorized_data, sigma=3.0, axis=0, maxiters=5
                )
            except TypeError:
                 _, median_winsorized, stddev_winsorized = sigma_clipped_stats_func(
                    winsorized_data, sigma_lower=3.0, sigma_upper=3.0, axis=0, maxiters=5
                )

            # _pcb("  Monochrome: Stats sur données winsorisées calculées.", lvl="DEBUG_VERY_DETAIL")

            lower_bound = median_winsorized - (sigma_low * stddev_winsorized)
            upper_bound = median_winsorized + (sigma_high * stddev_winsorized)
            
            pixels_to_reject = (original_data_NHW < lower_bound[np.newaxis, ...]) | \
                               (original_data_NHW > upper_bound[np.newaxis, ...])
                               
            rejection_mask_final = ~pixels_to_reject
            output_data_with_nans[pixels_to_reject] = np.nan
            
            num_rejected_mono = np.sum(pixels_to_reject)
            _pcb(f"  RejWinsor: Monochrome, {num_rejected_mono} pixels rejetés.", lvl="DEBUG_DETAIL")

    except MemoryError as e_mem:
        _pcb("reject_winsor_error_memory", lvl="ERROR", error=str(e_mem))
        _internal_logger.error("MemoryError dans _reject_outliers_winsorized_sigma_clip", exc_info=True)
        # En cas de MemoryError, il vaut mieux retourner les données originales pour ne pas planter
        return stacked_array_NHDWC.copy(), np.ones_like(stacked_array_NHDWC, dtype=bool)
    except Exception as e_winsor:
        _pcb("reject_winsor_error_unexpected", lvl="ERROR", error=str(e_winsor))
        _internal_logger.error("Erreur inattendue dans _reject_outliers_winsorized_sigma_clip", exc_info=True)
        return stacked_array_NHDWC.copy(), np.ones_like(stacked_array_NHDWC, dtype=bool)

    total_rejected_pixels = np.sum(~rejection_mask_final)
    _pcb("reject_winsor_info_finished", lvl="INFO_DETAIL", num_rejected=total_rejected_pixels)
    
    return output_data_with_nans, rejection_mask_final

def _reject_outliers_linear_fit_clip(
    stacked_array_NHDWC: np.ndarray,
    # Quels paramètres seraient nécessaires ?
    # Probablement des seuils pour le rejet des résidus,
    # peut-être des options pour le type de modèle à ajuster.
    # Pour l'instant, gardons-le simple.
    # sigma_clip_low_resid: float = 3.0, # Exemple de paramètre
    # sigma_clip_high_resid: float = 3.0, # Exemple de paramètre
    progress_callback: callable = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rejette les outliers en utilisant un Linear Fit Clipping (PLACEHOLDER).
    Cette méthode vise à modéliser et à soustraire les variations lentes (gradients)
    entre les images et l'image de référence (ex: médiane du stack), puis à rejeter
    les pixels qui s'écartent significativement de ce modèle.

    Args:
        stacked_array_NHDWC: Tableau des images empilées (N, H, W, C) ou (N, H, W).
        progress_callback: Fonction de callback pour les logs.
        // Ajouter d'autres paramètres au besoin.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - output_data_with_nans: Les données originales avec NaN où les pixels sont rejetés.
            - rejection_mask: Masque booléen (True où les pixels sont gardés).
    """
    _pcb = lambda msg_key, lvl="INFO_DETAIL", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else _internal_logger.debug(f"PCB_FALLBACK_{lvl}: {msg_key} {kwargs}")

    _pcb("reject_linearfit_warn_not_implemented", lvl="WARN",
         shape=stacked_array_NHDWC.shape)

    # Pour l'instant, cette fonction ne fait rien et retourne les données telles quelles.
    # L'implémentation réelle serait complexe.
    return stacked_array_NHDWC.copy(), np.ones_like(stacked_array_NHDWC, dtype=bool)


# zemosaic_align_stack.py

# ... (imports et autres fonctions restent les mêmes) ...

def stack_aligned_images(
    aligned_image_data_list: list[np.ndarray | None],
    normalize_method: str = 'none',
    weighting_method: str = 'none',
    rejection_algorithm: str = 'kappa_sigma',
    final_combine_method: str = 'mean',
    sigma_clip_low: float = 3.0,
    sigma_clip_high: float = 3.0,
    winsor_limits: tuple[float, float] = (0.05, 0.05),
    minimum_signal_adu_target: float = 0.0,
    apply_radial_weight: bool = False,
    radial_feather_fraction: float = 0.8,
    radial_shape_power: float = 2.0,
    winsor_max_workers: int = 1,
    progress_callback: callable = None
) -> np.ndarray | None:
    """
    Stacke une liste d'images alignées, appliquant normalisation, pondération (qualité + radiale),
    et rejet d'outliers optionnels. VERSION AVEC LOGS DE DEBUG INTENSIFS.
    ``winsor_max_workers`` permet de paralléliser la phase de Winsorisation lors
    du rejet Winsorized Sigma Clip.
    """
    _pcb = lambda msg_key, prog=None, lvl="INFO_DETAIL", **kwargs: \
        progress_callback(msg_key, prog, lvl, **kwargs) if progress_callback else _internal_logger.debug(f"PCB_FALLBACK_{lvl}_{prog}: {msg_key} {kwargs}")

    _pcb("STACK_IMG_ENTRY: Début stack_aligned_images.", lvl="ERROR") # Log d'entrée

    valid_images_to_stack = [img for img in aligned_image_data_list if img is not None and isinstance(img, np.ndarray)]
    if not valid_images_to_stack:
        _pcb("stackimages_warn_no_valid_images", lvl="WARN")
        _pcb("STACK_IMG_EXIT: Retourne None (pas d'images valides).", lvl="ERROR")
        return None

    num_images = len(valid_images_to_stack)
    _pcb("stackimages_info_start_stacking", lvl="INFO",
              num_images=num_images, norm=normalize_method,
              weight=weighting_method, reject=rejection_algorithm,
              combine=final_combine_method, 
              radial_weight_active=apply_radial_weight,
              radial_feather=radial_feather_fraction if apply_radial_weight else "N/A")

    # --- Préparation des images ---
    first_shape = None
    processed_images_for_stack = []
    for idx, img_adu in enumerate(valid_images_to_stack):
        current_img = img_adu 
        if current_img.dtype != np.float32:
            _pcb(f"StackImages: AVERT Image {idx} pas en float32 ({current_img.dtype}), conversion.", lvl="WARN")
            current_img = current_img.astype(np.float32, copy=True)
        
        # Vérification des infinités DÈS LE DÉBUT
        if not np.all(np.isfinite(current_img)):
            _pcb(f"STACK_IMG_PREP: AVERT Image {idx} (shape {current_img.shape}) contient des non-finis AVANT normalisation. Remplacement par 0.", lvl="ERROR")
            current_img = np.nan_to_num(current_img, nan=0.0, posinf=0.0, neginf=0.0)

        if first_shape is None: first_shape = current_img.shape
        elif current_img.shape != first_shape:
            _pcb("stackimages_warn_inconsistent_shape", lvl="WARN", img_index=idx, shape=current_img.shape, ref_shape=first_shape)
            continue 
        processed_images_for_stack.append(current_img)

    if not processed_images_for_stack:
        _pcb("stackimages_error_no_images_after_shape_check", lvl="ERROR")
        _pcb("STACK_IMG_EXIT: Retourne None (pas d'images après check shape).", lvl="ERROR")
        return None
    
    current_images_data_list = processed_images_for_stack # Renommage pour la suite
    _pcb(f"STACK_IMG_PREP: {len(current_images_data_list)} images prêtes pour normalisation.", lvl="ERROR")


    # --- NORMALISATION ---
    if normalize_method == 'linear_fit':
        _pcb("STACK_IMG_NORM: Appel _normalize_images_linear_fit.", lvl="ERROR")
        current_images_data_list = _normalize_images_linear_fit(current_images_data_list, progress_callback=progress_callback) # Params par défaut pour percentiles
    elif normalize_method == 'sky_mean':
        _pcb("STACK_IMG_NORM: Appel _normalize_images_sky_mean.", lvl="ERROR")
        current_images_data_list = _normalize_images_sky_mean(current_images_data_list, progress_callback=progress_callback)
    # ... (autres méthodes de normalisation si ajoutées) ...
    
    current_images_data_list = [img for img in current_images_data_list if img is not None] # Filtrer si normalisation a échoué pour certaines
    if not current_images_data_list:
        _pcb("stackimages_error_no_images_left_after_normalization_step", lvl="ERROR")
        _pcb("STACK_IMG_EXIT: Retourne None (pas d'images après normalisation).", lvl="ERROR")
        return None

    _pcb(f"STACK_IMG_NORM: {len(current_images_data_list)} images après normalisation. Vérification des non-finis POST-normalisation.", lvl="ERROR")
    temp_list_post_norm = []
    for idx_post_norm, img_post_norm in enumerate(current_images_data_list):
        if img_post_norm is not None:
            if not np.all(np.isfinite(img_post_norm)):
                _pcb(f"STACK_IMG_NORM: AVERT Image post-norm {idx_post_norm} (shape {img_post_norm.shape}) a des non-finis. Remplacement par 0.", lvl="ERROR")
                img_post_norm = np.nan_to_num(img_post_norm, nan=0.0, posinf=0.0, neginf=0.0)
            temp_list_post_norm.append(img_post_norm)
    current_images_data_list = temp_list_post_norm
    del temp_list_post_norm
    if not current_images_data_list: # Double check
        _pcb("STACK_IMG_NORM: Toutes les images sont devenues None après nettoyage post-normalisation.", lvl="ERROR")
        return None


    # --- PONDÉRATION DE QUALITÉ (Bruit/FWHM) ---
    quality_weights_list = None 
    _pcb(f"STACK_IMG_WEIGHT_QUAL: Début calcul poids qualité. Méthode: {weighting_method}", lvl="ERROR")
    if weighting_method == 'noise_variance':
        quality_weights_list = _calculate_image_weights_noise_variance(current_images_data_list, progress_callback=progress_callback)
    elif weighting_method == 'noise_fwhm':
        # ... (logique pour noise_fwhm, potentiellement combiner les deux) ...
        # Pour simplifier le debug actuel, si on veut "noise+fwhm", on pourrait avoir une clé dédiée
        # ou s'assurer que cette section calcule bien les deux et les combine.
        # Pour l'instant, elle calcule FWHM, puis Variance si FWHM a échoué ET si "variance" ou "noise" est dans la clé.
        # Testons d'abord avec 'none' pour la pondération qualité.
        if "fwhm" in weighting_method.lower():
             quality_weights_list = _calculate_image_weights_noise_fwhm(current_images_data_list, progress_callback=progress_callback)
        if ("variance" in weighting_method.lower() or "noise" in weighting_method.lower()) and \
           (quality_weights_list is None or not any(w is not None for w in quality_weights_list)): # Si fwhm a échoué ou n'a pas été demandé
            _pcb(f"STACK_IMG_WEIGHT_QUAL: Tentative calcul poids variance (soit demandé, soit FWHM a échoué).", lvl="ERROR")
            weights_var_temp = _calculate_image_weights_noise_variance(current_images_data_list, progress_callback=progress_callback)
            if quality_weights_list and weights_var_temp and any(w is not None for w in quality_weights_list) and any(w is not None for w in weights_var_temp):
                 _pcb(f"STACK_IMG_WEIGHT_QUAL: Combinaison poids FWHM et Variance.", lvl="ERROR")
                 # Assurer que les listes ont la même taille
                 len_q = len(quality_weights_list)
                 len_v = len(weights_var_temp)
                 # ... (logique de combinaison plus robuste si les longueurs diffèrent, ou erreur) ...
                 if len_q == len_v:
                     quality_weights_list = [ (w_f*w_v if w_f is not None and w_v is not None else (w_f if w_f is not None else w_v)) 
                                             for w_f, w_v in zip(quality_weights_list, weights_var_temp) ]
                 else: _pcb(f"STACK_IMG_WEIGHT_QUAL: ERREUR - Mismatch longueurs poids FWHM ({len_q}) et Variance ({len_v}).", lvl="ERROR")

            elif weights_var_temp:
                quality_weights_list = weights_var_temp
    # ... (autres méthodes de pondération qualité) ...
    _pcb(f"STACK_IMG_WEIGHT_QUAL: Fin calcul poids qualité. quality_weights_list is {'None' if quality_weights_list is None else 'Exists'}.", lvl="ERROR")
    if quality_weights_list and any(w is not None for w in quality_weights_list):
        # Trouver le premier poids non-None pour le log
        first_valid_q_weight = next((w for w in quality_weights_list if w is not None), None)
        if first_valid_q_weight is not None:
             _pcb(f"STACK_IMG_WEIGHT_QUAL: Premier quality_weight non-None - shape: {first_valid_q_weight.shape}, type: {first_valid_q_weight.dtype}, range: [{np.min(first_valid_q_weight):.3g}-{np.max(first_valid_q_weight):.3g}]", lvl="ERROR")


    # --- PONDÉRATION RADIALE ---
    final_radial_weights_list = [None] * len(current_images_data_list)
    _pcb(f"STACK_IMG_WEIGHT_RAD: Début calcul poids radiaux. Apply: {apply_radial_weight}", lvl="ERROR")
    if apply_radial_weight and ZEMOSAIC_UTILS_AVAILABLE_FOR_RADIAL and make_radial_weight_map_func:
        for idx, img_data_HWC in enumerate(current_images_data_list):
            if img_data_HWC is None: continue
            h, w = img_data_HWC.shape[:2]
            try:
                w_radial_2d = make_radial_weight_map_func(h, w, feather_fraction=radial_feather_fraction, shape_power=radial_shape_power)
                if img_data_HWC.ndim == 3:
                    final_radial_weights_list[idx] = np.repeat(w_radial_2d[..., np.newaxis], img_data_HWC.shape[-1], axis=2).astype(np.float32, copy=False)
                elif img_data_HWC.ndim == 2:
                    final_radial_weights_list[idx] = w_radial_2d.astype(np.float32, copy=False)
            except Exception as e_radw_post: # ... log erreur ...
                final_radial_weights_list[idx] = np.ones_like(img_data_HWC, dtype=np.float32)
    _pcb(f"STACK_IMG_WEIGHT_RAD: Fin calcul poids radiaux. final_radial_weights_list is {'None' if final_radial_weights_list is None else 'Exists'}.", lvl="ERROR")
    if final_radial_weights_list and any(w is not None for w in final_radial_weights_list):
        first_valid_r_weight = next((w for w in final_radial_weights_list if w is not None), None)
        if first_valid_r_weight is not None:
            _pcb(f"STACK_IMG_WEIGHT_RAD: Premier final_radial_weight non-None - shape: {first_valid_r_weight.shape}, type: {first_valid_r_weight.dtype}, range: [{np.min(first_valid_r_weight):.3g}-{np.max(first_valid_r_weight):.3g}]", lvl="ERROR")


    # --- COMBINAISON DES POIDS ---
    image_weights_list_combined = [None] * len(current_images_data_list)
    for i in range(len(current_images_data_list)):
        if current_images_data_list[i] is None: continue
        q_w = quality_weights_list[i] if quality_weights_list and i < len(quality_weights_list) and quality_weights_list[i] is not None else None
        r_w = final_radial_weights_list[i] if final_radial_weights_list[i] is not None else None
        if q_w is not None and r_w is not None: image_weights_list_combined[i] = q_w * r_w
        elif q_w is not None: image_weights_list_combined[i] = q_w
        elif r_w is not None: image_weights_list_combined[i] = r_w
        else: image_weights_list_combined[i] = None
    _pcb(f"STACK_IMG_WEIGHT_COMB: Poids combinés. image_weights_list_combined is {'None' if image_weights_list_combined is None else 'Exists'}.", lvl="ERROR")
    if image_weights_list_combined and any(w is not None for w in image_weights_list_combined):
        first_valid_c_weight = next((w for w in image_weights_list_combined if w is not None), None)
        if first_valid_c_weight is not None:
             _pcb(f"STACK_IMG_WEIGHT_COMB: Premier poids combiné non-None - shape: {first_valid_c_weight.shape}, type: {first_valid_c_weight.dtype}, range: [{np.min(first_valid_c_weight):.3g}-{np.max(first_valid_c_weight):.3g}]", lvl="ERROR")


    # --- STACKAGE NUMPY ---
    try:
        # S'assurer que current_images_data_list ne contient que des images valides avant stack
        valid_images_for_numpy_stack = [img for img in current_images_data_list if img is not None]
        if not valid_images_for_numpy_stack:
            _pcb("stackimages_error_no_images_to_stack_before_np_stack", lvl="ERROR")
            _pcb("STACK_IMG_EXIT: Retourne None (pas d'images avant np.stack).", lvl="ERROR")
            return None
            
        stacked_array_NHDWC = np.stack(valid_images_for_numpy_stack, axis=0)
        _pcb(f"STACK_IMG_NP_STACK: stacked_array_NHDWC - shape: {stacked_array_NHDWC.shape}, dtype: {stacked_array_NHDWC.dtype}, range: [{np.min(stacked_array_NHDWC):.2g}-{np.max(stacked_array_NHDWC):.2g}]", lvl="ERROR")

        # Filtrer les poids combinés pour correspondre EXACTEMENT aux images stackées
        filtered_combined_weights = [
            image_weights_list_combined[i] 
            for i, img in enumerate(current_images_data_list) # Itérer sur la liste originale avant filtrage pour garder les bons indices de poids
            if img is not None # Condition pour que l'image soit dans valid_images_for_numpy_stack
        ]
        del current_images_data_list, valid_images_for_numpy_stack # Libérer mémoire
        gc.collect()
    except Exception as e_np_stack:
        _pcb(f"stackimages_error_value_stacking_images: {e_np_stack}", lvl="ERROR")
        _pcb("STACK_IMG_EXIT: Retourne None (erreur np.stack).", lvl="ERROR")
        return None

    weights_array_NHDWC = None
    if filtered_combined_weights and any(w is not None for w in filtered_combined_weights):
        try:
            # Prendre uniquement les poids qui ne sont pas None
            valid_weights_to_stack_numpy = [w for w in filtered_combined_weights if w is not None]
            if not valid_weights_to_stack_numpy:
                _pcb("STACK_IMG_NP_STACK_WEIGHTS: Tous les poids filtrés sont None. weights_array_NHDWC sera None.", lvl="ERROR")
            elif len(valid_weights_to_stack_numpy) != stacked_array_NHDWC.shape[0]:
                 _pcb(f"STACK_IMG_NP_STACK_WEIGHTS: ERREUR - Mismatch nombre poids valides ({len(valid_weights_to_stack_numpy)}) et images stackées ({stacked_array_NHDWC.shape[0]}).", lvl="ERROR")
            else:
                weights_array_NHDWC = np.stack(valid_weights_to_stack_numpy, axis=0)
                if weights_array_NHDWC.shape != stacked_array_NHDWC.shape:
                    _pcb(f"stackimages_warn_combined_weights_shape_mismatch_final. Shape poids: {weights_array_NHDWC.shape}, Shape data: {stacked_array_NHDWC.shape}", lvl="ERROR")
                    weights_array_NHDWC = None 
        except Exception as e_w_stack:
            _pcb(f"stackimages_error_stacking_combined_weights: {e_w_stack}", lvl="ERROR")
            weights_array_NHDWC = None # S'assurer qu'il est None en cas d'erreur
    
    _pcb(f"STACK_IMG_NP_STACK_WEIGHTS: weights_array_NHDWC is {'None' if weights_array_NHDWC is None else 'Exists'}.", lvl="ERROR")
    if weights_array_NHDWC is not None:
         _pcb(f"STACK_IMG_NP_STACK_WEIGHTS: weights_array_NHDWC - shape: {weights_array_NHDWC.shape}, type: {weights_array_NHDWC.dtype}, range: [{np.min(weights_array_NHDWC):.3g}-{np.max(weights_array_NHDWC):.3g}]", lvl="ERROR")

    # ... (Nettoyage des listes de poids intermédiaires) ...
    del quality_weights_list, final_radial_weights_list, image_weights_list_combined, filtered_combined_weights
    gc.collect()

    # --- REJET D'OUTLIERS ---
    _pcb(f"STACK_IMG_REJECT: Début rejet. Algorithme: {rejection_algorithm}", lvl="ERROR")
    data_for_combine = stacked_array_NHDWC 
    rejection_mask = np.ones_like(stacked_array_NHDWC, dtype=bool) 
    if rejection_algorithm == 'kappa_sigma':
        data_for_combine, rejection_mask = _reject_outliers_kappa_sigma(stacked_array_NHDWC, sigma_clip_low, sigma_clip_high, progress_callback)
    elif rejection_algorithm == 'winsorized_sigma_clip':
        data_for_combine, rejection_mask = _reject_outliers_winsorized_sigma_clip(
            stacked_array_NHDWC,
            winsor_limits,
            sigma_clip_low,
            sigma_clip_high,
            progress_callback,
            winsor_max_workers,
        )
    # ... (autres algos de rejet) ...
    _pcb(f"STACK_IMG_REJECT: Fin rejet. data_for_combine shape: {data_for_combine.shape}, range: [{np.nanmin(data_for_combine):.2g}-{np.nanmax(data_for_combine):.2g}] (contient NaN)", lvl="ERROR")


    # --- COMBINAISON FINALE ---
    _pcb(f"STACK_IMG_COMBINE: Début combinaison finale. Méthode: {final_combine_method}", lvl="ERROR")
    result_image_adu = None
    try:
        effective_weights_for_combine = weights_array_NHDWC
        if effective_weights_for_combine is not None and rejection_mask is not None:
            _pcb(f"STACK_IMG_COMBINE: Application masque de rejet aux poids.", lvl="ERROR")
            effective_weights_for_combine = np.where(rejection_mask, weights_array_NHDWC, 0.0)
        
        _pcb(f"STACK_IMG_COMBINE: effective_weights_for_combine is {'None' if effective_weights_for_combine is None else 'Exists'}.", lvl="ERROR")
        if effective_weights_for_combine is not None:
            _pcb(f"STACK_IMG_COMBINE: effective_weights_for_combine - shape: {effective_weights_for_combine.shape}, dtype: {effective_weights_for_combine.dtype}, range: [{np.min(effective_weights_for_combine):.3g}-{np.max(effective_weights_for_combine):.3g}]", lvl="ERROR")


        if final_combine_method == 'mean':
            if effective_weights_for_combine is None:
                _pcb("STACK_IMG_COMBINE_MEAN: Pas de poids effectifs, utilisation np.nanmean.", lvl="ERROR")
                result_image_adu = np.nanmean(data_for_combine, axis=0)
            else:
                if data_for_combine.shape[0] != effective_weights_for_combine.shape[0]:
                    _pcb("stackimages_error_mean_combine_shape_mismatch (data vs effective_weights)", lvl="ERROR", 
                         data_N=data_for_combine.shape[0], weights_N=effective_weights_for_combine.shape[0])
                    result_image_adu = np.nanmean(data_for_combine, axis=0)
                else:
                    data_masked_for_avg = np.nan_to_num(data_for_combine, nan=0.0)
                    _pcb(f"STACK_IMG_COMBINE_MEAN: data_masked_for_avg - shape: {data_masked_for_avg.shape}, dtype: {data_masked_for_avg.dtype}, range: [{np.min(data_masked_for_avg):.2g}-{np.max(data_masked_for_avg):.2g}]", lvl="ERROR")
                    
                    if data_masked_for_avg.shape[0] == 1: # Cas N=1
                        _pcb(f"STACK_IMG_COMBINE_MEAN: N=1. Multiplication directe: data * poids_effectif.", lvl="ERROR")
                        # Log des pixels avant et après
                        img_idx_log, h_log, w_log, c_log = 0,0,0,0 # Pixel (0,0,0) de la première (unique) image
                        _pcb(f"  N=1 PRE-MULT: data_pixel=[{data_masked_for_avg[img_idx_log,h_log,w_log,c_log]:.3g}], weight_pixel=[{effective_weights_for_combine[img_idx_log,h_log,w_log,c_log]:.3g}]", lvl="ERROR")
                        result_image_adu = data_masked_for_avg[0] * effective_weights_for_combine[0]
                        _pcb(f"  N=1 POST-MULT: result_pixel=[{result_image_adu[h_log,w_log,c_log]:.3g}]", lvl="ERROR")
                    else: # Cas N > 1
                        _pcb(f"STACK_IMG_COMBINE_MEAN: N={data_masked_for_avg.shape[0]}. Moyenne pondérée standard.", lvl="ERROR")
                        sum_weighted_data = np.sum(data_masked_for_avg * effective_weights_for_combine, axis=0)
                        sum_weights = np.sum(effective_weights_for_combine, axis=0)
                        _pcb(f"  N>1 POST-SUM: sum_weighted_data range: [{np.min(sum_weighted_data):.2g}-{np.max(sum_weighted_data):.2g}]", lvl="ERROR")
                        _pcb(f"  N>1 POST-SUM: sum_weights range: [{np.min(sum_weights):.2g}-{np.max(sum_weights):.2g}]", lvl="ERROR")
                        
                        # Inspecter un pixel de bord pour sum_weights
                        bh,bw = 0,0 # Bord supérieur gauche
                        if sum_weights.ndim == 3 and sum_weights.shape[0]>bh and sum_weights.shape[1]>bw:
                             _pcb(f"  N>1 POST-SUM: sum_weights[{bh},{bw},0] = {sum_weights[bh,bw,0]:.3g}", lvl="ERROR")
                        
                        result_image_adu = np.divide(sum_weighted_data, sum_weights,
                                                   out=np.zeros_like(sum_weighted_data, dtype=np.float32),
                                                   where=sum_weights > 1e-9)
                        _pcb(f"  N>1 POST-DIVIDE: result_image_adu range: [{np.min(result_image_adu):.2g}-{np.max(result_image_adu):.2g}]", lvl="ERROR")
                        if result_image_adu.ndim == 3 and result_image_adu.shape[0]>bh and result_image_adu.shape[1]>bw:
                             _pcb(f"  N>1 POST-DIVIDE: result_image_adu[{bh},{bw},0] = {result_image_adu[bh,bw,0]:.3g}", lvl="ERROR")


        elif final_combine_method == 'median':
            # ... (logique pour median) ...
            if effective_weights_for_combine is not None:
                _pcb("stackimages_warn_median_with_weights_not_supported_simple", lvl="WARN")
            result_image_adu = np.nanmedian(data_for_combine, axis=0)
        # ... (else pour unknown combine method) ...

        if result_image_adu is not None and not np.all(np.isfinite(result_image_adu)):
            _pcb("STACK_IMG_COMBINE: AVERT - result_image_adu contient des non-finis POST-combinaison. Remplacement par 0.", lvl="ERROR")
            result_image_adu = np.nan_to_num(result_image_adu, nan=0.0, posinf=0.0, neginf=0.0)

    except Exception as e_comb_final:
        _pcb(f"stackimages_error_exception_final_combine: {e_comb_final}", lvl="ERROR")
        _internal_logger.error("Erreur combinaison finale dans stack_aligned_images", exc_info=True)
        _pcb("STACK_IMG_EXIT: Retourne None (erreur combinaison finale).", lvl="ERROR")
        return None
    finally:
        # ... (Nettoyage des gros tableaux) ...
        del data_for_combine, rejection_mask, stacked_array_NHDWC
        if weights_array_NHDWC is not None: del weights_array_NHDWC
        if 'effective_weights_for_combine' in locals() and effective_weights_for_combine is not None: del effective_weights_for_combine
        gc.collect()

    if result_image_adu is None:
        _pcb("stackimages_error_combine_result_none", lvl="ERROR")
        _pcb("STACK_IMG_EXIT: Retourne None (result_image_adu est None après try-except).", lvl="ERROR")
        return None

    _pcb(f"STACK_IMG_OFFSET: Avant offset final. Range: [{np.nanmin(result_image_adu):.2g}-{np.nanmax(result_image_adu):.2g}]", lvl="ERROR")
    # ... (Application de minimum_signal_adu_target) ...
    if result_image_adu is not None and minimum_signal_adu_target > 0.0:
        current_min_val = np.nanmin(result_image_adu)
        if np.isfinite(current_min_val) and current_min_val < minimum_signal_adu_target:
            offset_to_apply = minimum_signal_adu_target - current_min_val
            result_image_adu += offset_to_apply
    
    _pcb(f"STACK_IMG_EXIT: Fin stack_aligned_images. Range final: [{np.nanmin(result_image_adu):.2g}-{np.nanmax(result_image_adu):.2g}]", lvl="ERROR")
    return result_image_adu.astype(np.float32) if result_image_adu is not None else None





