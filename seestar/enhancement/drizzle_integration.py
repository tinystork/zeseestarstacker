# Standard library imports
import os
import traceback
import warnings
import gc
import logging

# Third party imports
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from reproject import reproject_exact

import inspect

import cv2
from scipy.ndimage import gaussian_filter
# ConvexHull n'est pas utilisé dans ce fichier

import warnings
import logging

_logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
#   Silence every "*is not a flux-conserving kernel*" warning that
#   originates from drizzle.resample.*  We ignore all repetitions.
# ------------------------------------------------------------------
warnings.filterwarnings(
    action="ignore",
    message=r".*is not a flux-conserving kernel\.$",
    module=r"drizzle\.resample"
)
_logger.debug("Flux-conserving kernel warnings are now filtered.")
# ------------------------------------------------------------------

logger = logging.getLogger(__name__)
# ConvexHull n'est pas utilisé dans ce fichier

try:
    import colour_demosaicing
    _DEBAYER_AVAILABLE = True
    logger.debug("DEBUG DrizzleIntegration: Found colour_demosaicing.")
except ImportError:
    _DEBAYER_AVAILABLE = False
    logger.warning("WARNING DrizzleIntegration: colour-demosaicing library not found.")
    class colour_demosaicing: # Factice
        @staticmethod
        def demosaicing_CFA_Bayer_Malvar2004(data, pattern):
            logger.error("ERROR: colour_demosaicing not available for demosaicing.")
            return data


# try: # Pas besoin de SeestarQueuedStacker ici
#     from ..queuep.queue_manager import SeestarQueuedStacker 
# except ImportError:
#     SeestarQueuedStacker = None 

warnings.filterwarnings('ignore', category=FITSFixedWarning)

# --- Fonctions Helper de Module ---
def _load_drizzle_temp_file(filepath): # Nom corrigé
    try:
        with fits.open(filepath, memmap=False) as hdul:
            if not hdul or not hdul[0].is_image or hdul[0].data is None: 
                # logger.debug(f"DEBUG _load_drizzle_temp_file: HDU invalide pour {filepath}")
                return None, None, None
            hdu = hdul[0]; data = hdu.data; header = hdu.header
            data_hxwx3 = None
            if data.ndim == 3:
                if data.shape[2] == 3: data_hxwx3 = data.astype(np.float32)
                elif data.shape[0] == 3: data_hxwx3 = np.moveaxis(data, 0, -1).astype(np.float32)
                else: 
                    # logger.debug(f"DEBUG _load_drizzle_temp_file: Shape 3D inattendue {data.shape} pour {filepath}")
                    return None, None, None
            else: 
                # logger.debug(f"DEBUG _load_drizzle_temp_file: Données non 3D {data.ndim}D pour {filepath}")
                return None, None, None
            wcs = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', FITSFixedWarning)
                    wcs_hdr = WCS(header, naxis=2)
                if wcs_hdr.is_celestial: wcs = wcs_hdr
            except Exception: pass
            if wcs is None: 
                # logger.debug(f"DEBUG _load_drizzle_temp_file: WCS non trouvé/valide pour {filepath}")
                return None, None, None
            return data_hxwx3, wcs, header
    except FileNotFoundError: 
        # logger.debug(f"DEBUG _load_drizzle_temp_file: Fichier non trouvé {filepath}")
        return None, None, None
    except Exception as e: 
        logger.debug(f"ERREUR _load_drizzle_temp_file pour {filepath}: {e}")
        traceback.print_exc(limit=1); return None, None, None

def _create_wcs_from_header(header): # Nom corrigé
    required_keys = ['NAXIS1', 'NAXIS2', 'RA', 'DEC', 'FOCALLEN', 'XPIXSZ', 'YPIXSZ']
    if not all(key in header for key in required_keys): 
        # logger.debug(f"DEBUG _create_wcs_from_header: Clés manquantes { [k for k in required_keys if k not in header] }")
        return None
    try:
        naxis1 = int(header['NAXIS1']); naxis2 = int(header['NAXIS2'])
        ra_deg = float(header['RA']); dec_deg = float(header['DEC'])
        focal_len_mm = float(header['FOCALLEN'])
        pixel_size_x_um = float(header['XPIXSZ']); pixel_size_y_um = float(header['YPIXSZ'])
        # Conversion en mètres pour la formule d'échelle
        focal_len_m = focal_len_mm * 1e-3
        pixel_size_x_m = pixel_size_x_um * 1e-6; pixel_size_y_m = pixel_size_y_um * 1e-6
        # Échelle en radians/pixel
        scale_x_rad_per_pix = pixel_size_x_m / focal_len_m
        scale_y_rad_per_pix = pixel_size_y_m / focal_len_m
        # Conversion en degrés/pixel
        deg_per_rad = 180.0 / np.pi
        scale_x_deg_per_pix = scale_x_rad_per_pix * deg_per_rad
        scale_y_deg_per_pix = scale_y_rad_per_pix * deg_per_rad

        w = WCS(naxis=2)
        # Point de référence au centre de l'image (convention FITS 1-based)
        w.wcs.crpix = [naxis1 / 2.0 + 0.5, naxis2 / 2.0 + 0.5]
        w.wcs.crval = [ra_deg, dec_deg] # Coordonnées célestes au point de référence
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"] # Type de projection (tangentielle)
        # Échelle en degrés par pixel. Négatif pour RA car RA augmente vers la gauche.
        w.wcs.cdelt = np.array([-scale_x_deg_per_pix, scale_y_deg_per_pix]) 
        w.wcs.cunit = ['deg', 'deg'] # Unités des cdelt et crval
        w.wcs.pc = np.array([[1.0, 0.0], [0.0, 1.0]]) # Matrice de rotation (identité ici)
        return w
    except Exception as e_wcs_create: 
        logger.debug(f"ERREUR _create_wcs_from_header: {e_wcs_create}")
        return None

def _load_fits_data_wcs_debayered(filepath, bayer_pattern='GRBG'): # Nom corrigé
    # logger.debug(f"DEBUG _load_fits_data_wcs_debayered: Tentative chargement {filepath}")
    try:
        with fits.open(filepath, memmap=False) as hdul:
            hdu = None; header = None
            for h_item in hdul:
                if h_item.is_image and h_item.data is not None and h_item.data.ndim == 2:
                    hdu = h_item; break
            if hdu is None: 
                # logger.debug(f"DEBUG _load_fits_data_wcs_debayered: Aucune HDU 2D image valide dans {filepath}")
                return None, None, None
            
            bayer_data = hdu.data; header = hdu.header
            rgb_image = None
            if _DEBAYER_AVAILABLE and colour_demosaicing is not None: # Vérifier aussi que l'objet existe
                try:
                    bayer_float = bayer_data.astype(np.float32)
                    valid_patterns = ['GRBG', 'RGGB', 'GBRG', 'BGGR'] # Assurer que le pattern est valide
                    pattern_upper = bayer_pattern.upper()
                    if pattern_upper not in valid_patterns: pattern_upper = 'GRBG'
                    
                    rgb_image = colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004(bayer_float, pattern=pattern_upper)
                    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3: 
                        raise ValueError(f"Debayer a retourné une shape inattendue: {rgb_image.shape}")
                except Exception as e_debayer:
                    logger.debug(f"ERREUR Debayer dans _load_fits_data_wcs_debayered pour {filepath}: {e_debayer}")
                    return None, None, None # Échec du debayering
            else: 
                logger.debug(f"WARN _load_fits_data_wcs_debayered: Bibliothèque colour_demosaicing non dispo pour {filepath}.")
                return None, None, None # Pas de debayering possible
            
            wcs = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', FITSFixedWarning)
                    wcs_hdr = WCS(header, naxis=2)
                if wcs_hdr.is_celestial: wcs = wcs_hdr
            except Exception: pass

            if wcs is None: # Si WCS du header échoue, essayer de générer
                wcs_gen = _create_wcs_from_header(header) # Appel au helper de module corrigé
                if wcs_gen and wcs_gen.is_celestial: wcs = wcs_gen
            
            if not wcs: 
                # logger.debug(f"DEBUG _load_fits_data_wcs_debayered: WCS non trouvé/généré pour {filepath}")
                return None, None, None # WCS est essentiel
            
            return rgb_image.astype(np.float32), wcs, header
            
    except FileNotFoundError: 
        # logger.debug(f"DEBUG _load_fits_data_wcs_debayered: Fichier non trouvé {filepath}")
        return None, None, None
    except Exception as e_load: 
        logger.debug(f"ERREUR _load_fits_data_wcs_debayered pour {filepath}: {e_load}")
        traceback.print_exc(limit=1); return None, None, None

class DrizzleIntegrator:
    """Combine drizzle outputs while optionally renormalizing the flux."""

    def __init__(self, renormalize: str = "max") -> None:
        self.renormalize = renormalize
        self._sci_accum: np.ndarray | None = None
        self._wht_accum: np.ndarray | None = None
        self._n_images = 0

    def add(self, sci: np.ndarray, wht: np.ndarray) -> None:
        """Add a drizzle result to the accumulation."""
        sci = np.asarray(sci, dtype=np.float32)
        wht = np.asarray(wht, dtype=np.float32)
        if self._sci_accum is None:
            self._sci_accum = sci.copy()
            self._wht_accum = wht.copy()
        else:
            self._sci_accum += sci
            self._wht_accum += wht
        self._n_images += 1

    def current_preview(self) -> np.ndarray:
        """Return a float32 normalised copy of the current stack.

        Returns:
            np.ndarray: Normalised drizzle accumulation.

        Raises:
            ValueError: If no images were added yet.
        """
        if self._sci_accum is None or self._wht_accum is None:
            raise ValueError("No images added")

        return (self._sci_accum / np.maximum(self._wht_accum, 1e-9)).astype(
            np.float32
        )

    def cumulative_preview(self) -> np.ndarray:
        """Return the current drizzle stack normalised to ``[0,1]``.

        The returned array shares no memory with the internal accumulators and
        is always ``float32``. The integrator state is unchanged.

        Returns
        -------
        np.ndarray
            Normalised cumulative drizzle image.
        """

        if self._sci_accum is None or self._wht_accum is None:
            raise ValueError("No images added")

        return (self._sci_accum / np.maximum(self._wht_accum, 1e-9)).astype(
            np.float32, copy=False
        )

    def finalize(self) -> np.ndarray:
        """Return the stacked image after optional renormalization."""
        if self._sci_accum is None or self._wht_accum is None:
            raise ValueError("No images added")

        wht_accum_safe = np.maximum(self._wht_accum, 1e-9)
        final_img = self._sci_accum / wht_accum_safe
        if self.renormalize == "max":
            final_img *= wht_accum_safe.max()
        elif self.renormalize == "n_images":
            final_img *= self._n_images
        return final_img.astype(np.float32)


def run_incremental_drizzle(
    images,
    wcs_list,
    target_wcs,
    shape_out,
    *,
    pixfrac=1.0,
    kernel="square",
):
    """Stack images incrementally using ``reproject_exact``.

    Parameters
    ----------
    images : iterable of `numpy.ndarray`
        2D image arrays to drizzle.
    wcs_list : iterable of `astropy.wcs.WCS`
        Input WCS corresponding to each image.
    target_wcs : `astropy.wcs.WCS`
        Output WCS defining the drizzle grid.
    shape_out : tuple
        Shape ``(ny, nx)`` of the output grid.
    pixfrac : float, optional
        Drizzle ``pixfrac`` parameter.
    kernel : str, optional
        Drizzle kernel.

    Returns
    -------
    numpy.ndarray
        The drizzled image.
    """

    sum_data = np.zeros(shape_out, dtype=float)
    sum_weight = np.zeros(shape_out, dtype=float)

    drizzle_kwargs = {}
    if "drizzle" in inspect.signature(reproject_exact).parameters:
        drizzle_kwargs = {"drizzle": True, "pixfrac": pixfrac, "kernel": kernel}

    for data, wcs_in in zip(images, wcs_list):
        arr, fp = reproject_exact(
            (data, wcs_in),
            target_wcs,
            shape_out=shape_out,

            **drizzle_kwargs,

        )
        sum_data += arr * fp
        sum_weight += fp

    valid = sum_weight > 0
    final = np.zeros_like(sum_data)
    final[valid] = sum_data[valid] / sum_weight[valid]
    return final


def run_final_drizzle(
    images,
    wcs_list,
    final_target_wcs,
    final_shape_out,
    *,
    pixfrac=1.0,
    kernel="square",
):
    """Combine all aligned images using ``reproject_exact``.

    Parameters
    ----------
    images : iterable of `numpy.ndarray`
        Aligned image arrays.
    wcs_list : iterable of `astropy.wcs.WCS`
        WCS for each image.
    final_target_wcs : `astropy.wcs.WCS`
        Output WCS for the final drizzle.
    final_shape_out : tuple
        Output array shape ``(ny, nx)``.
    pixfrac : float, optional
        Drizzle ``pixfrac`` parameter.
    kernel : str, optional
        Drizzle kernel.

    Returns
    -------
    numpy.ndarray
        Final drizzled image.
    """


    drizzle_kwargs = {}
    if "drizzle" in inspect.signature(reproject_exact).parameters:
        drizzle_kwargs = {"drizzle": True, "pixfrac": pixfrac, "kernel": kernel}


    sum_data = np.zeros(final_shape_out, dtype=float)
    sum_weight = np.zeros(final_shape_out, dtype=float)
    for data, wcs_in in zip(images, wcs_list):
        arr, fp = reproject_exact(
            (data, wcs_in),
            final_target_wcs,
            shape_out=final_shape_out,

            **drizzle_kwargs,

        )
        sum_data += arr * fp
        sum_weight += fp

    valid = sum_weight > 0
    final = np.zeros_like(sum_data)
    final[valid] = sum_data[valid] / sum_weight[valid]
    return final


# --- FIN DU FICHIER seestar/enhancement/drizzle_integration.py ---