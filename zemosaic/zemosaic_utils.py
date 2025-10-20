# zemosaic_utils.py

# --- Standard Library Imports ---
import os
import numpy as np
# L'import de astropy.io.fits est géré ci-dessous pour définir le flag
import cv2

import warnings
import traceback
import gc
import importlib.util

# --- GPU/CUDA Availability ----------------------------------------------------
GPU_AVAILABLE = importlib.util.find_spec("cupy") is not None
map_coordinates = None  # Lazily imported when needed

from reproject.mosaicking import reproject_and_coadd as cpu_reproject_and_coadd

# --- Définition locale du flag ASTROPY_AVAILABLE et du module fits pour ce fichier ---
ASTROPY_AVAILABLE_IN_UTILS = False
fits_module_for_utils = None # Contiendra le module fits réel ou un mock

# IMPORTS D'ASTROPY POUR LA VISUALISATION
ASTROPY_VISUALIZATION_AVAILABLE = False
ImageNormalize, PercentileInterval, AsinhStretch, LogStretch = None, None, None, None # Pour les type hints et fallback
try:
    from astropy.visualization import (ImageNormalize, PercentileInterval, 
                                       AsinhStretch, LogStretch) # Et d'autres si besoin (SqrtStretch, etc.)
    ASTROPY_VISUALIZATION_AVAILABLE = True
    # print("INFO (zemosaic_utils): astropy.visualization importé.")
except ImportError:
    print("AVERT (zemosaic_utils): astropy.visualization non disponible. L'étirement asinh avancé ne sera pas possible.")

try:
    from astropy.io import fits as actual_fits_for_utils
    from astropy.io.fits.verify import VerifyWarning # Importer ici si Astropy est là
    warnings.filterwarnings("ignore", category=VerifyWarning, message="Keyword name.*is greater than 8 characters.*")
    warnings.filterwarnings("ignore", category=VerifyWarning, message="Keyword name.*contains characters not allowed.*")
    fits_module_for_utils = actual_fits_for_utils
    ASTROPY_AVAILABLE_IN_UTILS = True
    # print("INFO (zemosaic_utils): Astropy (fits) importé avec succès pour ce module.")
except ImportError:
    # Créer un placeholder minimal pour fits si Astropy n'est pas là du tout
    class MockFitsCard: 
        def __init__(self, key, value, comment=''):
            self.keyword = key; self.value = value; self.comment = comment
    class MockFitsHeader:
        def __init__(self): self._cards = {}; self.comments = {}
        def update(self, other):
            if isinstance(other, MockFitsHeader): self._cards.update(other._cards); self.comments.update(other.comments)
            elif isinstance(other, dict):
                for k, v_tuple in other.items():
                    if isinstance(v_tuple, tuple) and len(v_tuple) == 2: self._cards[k] = v_tuple[0]; self.comments[k] = v_tuple[1]
                    else: self._cards[k] = v_tuple
        def copy(self): new_header = MockFitsHeader(); new_header._cards = self._cards.copy(); new_header.comments = self.comments.copy(); return new_header
        def __contains__(self, key): return key in self._cards
        def __delitem__(self, key):
            if key in self._cards: del self._cards[key]
            if key in self.comments: del self.comments[key]
        def __setitem__(self, key, value_comment_tuple):
            if isinstance(value_comment_tuple, tuple) and len(value_comment_tuple) == 2: self._cards[key] = value_comment_tuple[0]; self.comments[key] = value_comment_tuple[1]
            else: self._cards[key] = value_comment_tuple
        def get(self, key, default=None): return self._cards.get(key, default)
        def cards(self):
            for k, v in self._cards.items(): yield MockFitsCard(k,v,self.comments.get(k,''))
  
    class MockPrimaryHDU: # Renommé pour éviter conflit si MockHDU est aussi défini
        def __init__(self, data=None, header=None):
            self.data = data
            self.header = header if header is not None else MockFitsHeader()
            self.is_image = data is not None 
            self.shape = data.shape if hasattr(data, 'shape') else None
            self.name = "PRIMARY"
        def copy(self): return MockPrimaryHDU(self.data.copy() if self.data is not None else None, self.header.copy())
  
    class MockHDU: 
        def __init__(self, data=None, header=None):
            self.data = data
            self.header = header if header is not None else MockFitsHeader()
            self.is_image = data is not None
            self.shape = data.shape if hasattr(data, 'shape') else None
            self.name = "PRIMARY"
        def copy(self): return MockHDU(self.data.copy() if self.data is not None else None, self.header.copy())

    class MockHDUList:
        def __init__(self, hdus=None):
            self.hdus = hdus if hdus is not None else []
        def __getitem__(self, key): return self.hdus[key]
        def __len__(self): return len(self.hdus)
        def writeto(self, output_path, overwrite=True, checksum=False, output_verify='fix'): pass
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): self.close()

    class MockFitsModule:
        Header = MockFitsHeader
        PrimaryHDU = MockPrimaryHDU
        HDUList = MockHDUList
        @staticmethod
        def open(filepath, memmap=False, do_not_scale_image_data=True):
            print(f"MOCK fits_module_for_utils.open CALLED for {filepath} (Astropy not found). Returning minimal mock HDU.")
            # Simuler une HDU minimale pour éviter des crashs dans load_and_validate_fits
            # Ce mock est très basique et ne lira pas réellement le fichier.
            mock_data = np.array([[0,0],[0,0]], dtype=np.int16) # Petite donnée pour avoir un .shape
            mock_header = MockFitsHeader()
            mock_header['NAXIS'] = 2
            mock_header['NAXIS1'] = 2
            mock_header['NAXIS2'] = 2
            return MockHDUList([MockHDU(data=mock_data, header=mock_header)])
        @staticmethod
        def getheader(filepath, ext=0):
            print(f"MOCK fits_module_for_utils.getheader CALLED for {filepath} (Astropy not found).")
            return MockFitsHeader()
    
    fits_module_for_utils = MockFitsModule()
    print("AVERTISSEMENT (zemosaic_utils): Astropy (fits) non trouvé. Fonctionnalités FITS limitées/mockées.")
# --- Fin Définition locale ---

warnings.filterwarnings("ignore", category=FutureWarning)
# Les filtres VerifyWarning sont maintenant dans le try/except d'Astropy ci-dessus.







# DANS zemosaic_utils.py

# (Les imports et la définition de ASTROPY_AVAILABLE_IN_UTILS, fits_module_for_utils restent les mêmes)
# ...

def load_and_validate_fits(filepath,
                           normalize_to_float32=True, # Si True, normalise la sortie à [0,1]
                           attempt_fix_nonfinite=True,
                           progress_callback=None):
    filename = os.path.basename(filepath)

    def _log_util(message, level="DEBUG_DETAIL"):
        if progress_callback and callable(progress_callback):
            progress_callback(f"  [ZMU_LoadVal] {message}", None, level)
        else:
            print(f"  [ZMU_LoadVal PRINTFALLBACK] {level}: {message}")

    _log_util(f"Début chargement (V3 - BZERO/BSCALE affiné). Fichier: '{filename}'. NormalizeOutput01={normalize_to_float32}, FixNonFinite={attempt_fix_nonfinite}", "DEBUG")

    data_raw_from_fits = None  # Données telles que lues par fits.open
    header = None
    header_for_fallback = fits_module_for_utils.Header()
    info = {}

    try:
        _log_util(f"Tentative fits_module_for_utils.open('{filepath}', do_not_scale_image_data=True)...", "DEBUG_DETAIL")
        with fits_module_for_utils.open(filepath, memmap=False, do_not_scale_image_data=True) as hdul:
            # ... (logique de sélection de hdu_img inchangée) ...
            _log_util(f"fits_module_for_utils.open OK. Nombre HDUs: {len(hdul) if hdul else 0}", "DEBUG_DETAIL")
            if not hdul:
                _log_util(f"REJET: Fichier FITS vide ou corrompu (hdul est None/vide).", "WARN")
                return None, header_for_fallback, info

            hdu_img = None; img_hdu_idx = -1
            _log_util(f"Recherche HDU image...", "DEBUG_DETAIL")
            for idx, hdu_item in enumerate(hdul):
                is_image_attr = getattr(hdu_item, 'is_image', False)
                has_data_attr = hasattr(hdu_item, 'data')
                if is_image_attr and has_data_attr and hdu_item.data is not None:
                    _log_util(f"  HDU {idx} est image. Shape brute: {hdu_item.data.shape if hasattr(hdu_item.data, 'shape') else 'N/A'}, Dtype brut: {hdu_item.data.dtype if hasattr(hdu_item.data, 'dtype') else 'N/A'}", "DEBUG_DETAIL")
                    hdu_name = getattr(hdu_item, 'name', 'N/A_NAME')
                    if idx == 0 or (isinstance(hdu_name, str) and hdu_name.upper() in ['SCI', 'IMAGE', 'PRIMARY']):
                        hdu_img = hdu_item; img_hdu_idx = idx
                        _log_util(f"  HDU prioritaire {img_hdu_idx} ('{hdu_name}') sélectionnée.", "DEBUG_DETAIL"); break
            if hdu_img is None:
                _log_util(f"Pas de HDU prioritaire. Recherche première HDU image...", "DEBUG_DETAIL")
                for idx, hdu_item in enumerate(hdul):
                    is_image_attr = getattr(hdu_item, 'is_image', False); has_data_attr = hasattr(hdu_item, 'data')
                    if is_image_attr and has_data_attr and hdu_item.data is not None:
                        hdu_img = hdu_item; img_hdu_idx = idx
                        _log_util(f"  Première HDU image {img_hdu_idx} sélectionnée.", "DEBUG_DETAIL"); break
            if hdu_img is None or hdu_img.data is None:
                _log_util(f"REJET: Aucune HDU image valide avec données.", "WARN")
                if ASTROPY_AVAILABLE_IN_UTILS and len(hdul) > 0 and hasattr(hdul[0], 'header') and hdul[0].header:
                    header_for_fallback = hdul[0].header.copy()
                return None, header_for_fallback, info

            data_raw_from_fits = hdu_img.data # Peut être int16, uint16, float32, etc.
            header = hdu_img.header.copy(); header_for_fallback = header.copy()
            
            _log_util(f"Données lues HDU {img_hdu_idx}. Shape brute: {data_raw_from_fits.shape}, Dtype brut: {data_raw_from_fits.dtype}", "DEBUG")
            _log_util(f"  Range brut (depuis FITS): [{np.min(data_raw_from_fits) if data_raw_from_fits.size>0 else 'N/A'}, {np.max(data_raw_from_fits) if data_raw_from_fits.size>0 else 'N/A'}]", "DEBUG")

            # --- Conversion en float64 pour le scaling ADU et autres opérations ---
            data_scaled_f64 = data_raw_from_fits.astype(np.float64)
            _log_util(f"Converti en float64. Range: [{np.min(data_scaled_f64):.1f}, {np.max(data_scaled_f64):.1f}]", "DEBUG_DETAIL")

            # --- Application BZERO/BSCALE si `do_not_scale_image_data=True` a empêché Astropy ---
            # Et si les données d'origine étaient des entiers.
            if data_raw_from_fits.dtype.kind == 'i': 
                bzero = header.get('BZERO', 0.0)
                bscale = header.get('BSCALE', 1.0)
                if abs(bscale - 1.0) > 1e-6 or abs(bzero) > 1e-6:
                    _log_util(f"Application BZERO={bzero}, BSCALE={bscale} (car do_not_scale_image_data=True et dtype entier).", "INFO_DETAIL")
                    # data_scaled_f64 était déjà une copie de data_raw_from_fits en float64
                    data_scaled_f64 = data_scaled_f64 * bscale + bzero 
                    _log_util(f"  Après BZERO/BSCALE: Range [{np.min(data_scaled_f64):.1f}, {np.max(data_scaled_f64):.1f}], Dtype: {data_scaled_f64.dtype}", "DEBUG")
                else:
                    _log_util(f"BZERO/BSCALE triviaux ou absents pour dtype entier. Pas de scaling manuel BZ/BS.", "DEBUG_DETAIL")
            else:
                 _log_util(f"Dtype brut ({data_raw_from_fits.dtype}) non entier. Pas de scaling BZERO/BSCALE manuel appliqué.", "DEBUG_DETAIL")
            # À ce stade, data_scaled_f64 devrait être en float64 et avoir la bonne plage ADU (ex: 0-65535 pour un Seestar FITS)

            # --- Transposition si nécessaire ---
            data_transposed_f64 = data_scaled_f64
            axis_orig = "HWC"
            if data_scaled_f64.ndim == 3:
                if data_scaled_f64.shape[0] in [1, 3, 4] and data_scaled_f64.shape[1] > 4 and data_scaled_f64.shape[2] > 4:
                    _log_util(f"Shape 3D {data_scaled_f64.shape} type CxHxW. Transposition vers HxWxC...", "INFO_DETAIL")
                    data_transposed_f64 = np.moveaxis(data_scaled_f64, 0, -1)
                    axis_orig = "CHW"
                    _log_util(f"  Shape après transposition: {data_transposed_f64.shape}", "DEBUG_DETAIL")
                elif data_scaled_f64.shape[2] in [1, 3, 4] and data_scaled_f64.shape[0] > 4 and data_scaled_f64.shape[1] > 4:
                    _log_util(f"Shape 3D {data_scaled_f64.shape} déjà HxWxC.", "DEBUG_DETAIL")
                    axis_orig = "HWC"
                else:
                    _log_util(f"REJET: Shape 3D non supportée ({data_scaled_f64.shape}).", "WARN"); return None, header, info
            elif data_scaled_f64.ndim != 2:
                _log_util(f"REJET: Shape {data_scaled_f64.ndim}D non supportée.", "WARN"); return None, header, info

            info["axis_order_original"] = axis_orig
            
            _log_util(f"Après transposition (si 3D): Range [{np.min(data_transposed_f64):.1f}, {np.max(data_transposed_f64):.1f}], Dtype: {data_transposed_f64.dtype}", "DEBUG")
            
            # --- Gestion NaN/Inf ---
            data_cleaned_f64 = data_transposed_f64
            if attempt_fix_nonfinite:
                if not np.all(np.isfinite(data_transposed_f64)):
                    _log_util(f"AVERT: Données non finies détectées. Remplacement par 0.0.", "WARN")
                    data_cleaned_f64 = np.nan_to_num(data_transposed_f64, nan=0.0, posinf=0.0, neginf=0.0)
                    _log_util(f"  Après nan_to_num: Range [{np.min(data_cleaned_f64):.1f}, {np.max(data_cleaned_f64):.1f}]", "DEBUG_DETAIL")
            
            # --- Conversion finale en float32 pour la sortie ---
            image_data_final_float32 = data_cleaned_f64.astype(np.float32)
            _log_util(f"Converti en float32 final. Range: [{np.min(image_data_final_float32):.3g}, {np.max(image_data_final_float32):.3g}]", "DEBUG")

            # --- Normalisation optionnelle à [0,1] ---
            if normalize_to_float32:
                _log_util(f"Normalisation 0-1 demandée...", "DEBUG_DETAIL")
                min_val_norm, max_val_norm = np.nanmin(image_data_final_float32), np.nanmax(image_data_final_float32)
                _log_util(f"  Min/Max pour normalisation 0-1: [{min_val_norm:.3g}, {max_val_norm:.3g}]", "DEBUG_DETAIL")
                
                if np.isfinite(min_val_norm) and np.isfinite(max_val_norm) and (max_val_norm > min_val_norm + 1e-9):
                    image_data_final_float32 = (image_data_final_float32 - min_val_norm) / (max_val_norm - min_val_norm)
                    image_data_final_float32 = np.clip(image_data_final_float32, 0.0, 1.0)
                elif np.any(np.isfinite(image_data_final_float32)): # Image constante non-Nan/Inf
                    image_data_final_float32 = np.full_like(image_data_final_float32, 0.5, dtype=np.float32)
                    _log_util(f"  Image constante, normalisée à 0.5.", "DEBUG_DETAIL")
                else: # Tout NaN ou Inf
                    image_data_final_float32 = np.zeros_like(image_data_final_float32, dtype=np.float32)
                    _log_util(f"  Image non-finie, normalisée à 0.0.", "DEBUG_DETAIL")
                _log_util(f"Normalisation 0-1 effectuée. Range après: [{np.nanmin(image_data_final_float32):.3f}, {np.nanmax(image_data_final_float32):.3f}]", "DEBUG_DETAIL")
            else:
                _log_util(f"Pas de normalisation 0-1 (ADU). Range final: [{np.nanmin(image_data_final_float32):.3g}, {np.nanmax(image_data_final_float32):.3g}]", "DEBUG_DETAIL")

            _log_util(
                f"FIN chargement '{filename}'. Shape: {image_data_final_float32.shape}, Dtype: {image_data_final_float32.dtype}, "
                f"Range: [{np.nanmin(image_data_final_float32):.3g} - {np.nanmax(image_data_final_float32):.3g}], Mean: {np.nanmean(image_data_final_float32):.3g}",
                "INFO",
            )
            return image_data_final_float32, header, info

    except FileNotFoundError:
        _log_util(f"ERREUR CRITIQUE: Fichier non trouvé: '{filepath}'", "ERROR")
        return None, header_for_fallback, info
    except MemoryError as me:
        _log_util(f"ERREUR CRITIQUE MÉMOIRE: {me}", "ERROR"); return None, header_for_fallback, info
    except Exception as e:
        _log_util(f"ERREUR INATTENDUE chargement/validation '{filename}': {type(e).__name__} - {e}", "ERROR")
        if progress_callback and hasattr(progress_callback.__self__ if hasattr(progress_callback, '__self__') else progress_callback, 'logger'):
             logger_instance = progress_callback.__self__.logger if hasattr(progress_callback, '__self__') else _log_util.getLogger("ZeMosaicUtilsUnknownContext")
             logger_instance.error(f"Traceback pour load_and_validate_fits (fichier: {filename}):", exc_info=True)
        elif progress_callback:
             progress_callback(f"  [ZMU_LoadVal TRACEBACK] {traceback.format_exc(limit=3)}", None, "ERROR")
        else:
            traceback.print_exc(limit=3)
        return None, header_for_fallback, info





def crop_image_and_wcs(
    image_data_hwc: np.ndarray, 
    wcs_obj, # Type hint générique pour éviter les problèmes avec Pylance si WCS n'est pas toujours AstropyWCSBase
    crop_percentage_per_side: float,
    progress_callback: callable = None
) -> tuple[np.ndarray | None, object | None]: # Type de retour générique pour WCS
    """
    Rogne une image (HWC ou HW) d'un certain pourcentage sur chaque côté
    et ajuste l'objet WCS Astropy correspondant.

    Args:
        image_data_hwc (np.ndarray): Tableau image (H, W, C) ou (H, W).
        wcs_obj (astropy.wcs.WCS): Objet WCS original.
        crop_percentage_per_side (float): Fraction (0.0 à <0.5) à rogner de chaque côté.
        progress_callback (callable, optional): Fonction pour les logs.

    Returns:
        tuple: (cropped_image_data, cropped_wcs_obj) ou (None, None) si erreur,
               ou (image_data_hwc, wcs_obj) si pas de rognage ou pas d'Astropy.
    """
    # Définir un logger local simple pour cette fonction si _internal_logger n'est pas souhaité/disponible
    # ou utiliser progress_callback pour tous les messages.
    # Pour simplifier, j'utilise progress_callback s'il est fourni.
    def _pcb_crop(message, level="DEBUG_DETAIL", **kwargs):
        if progress_callback and callable(progress_callback):
            # Préfixer pour identifier l'origine du log si besoin
            progress_callback(f"[CropUtil] {message}", None, level, **kwargs)
        else:
            # Fallback simple si pas de callback (pourrait arriver si utilisé ailleurs)
            print(f"CROP_UTIL_LOG {level}: {message} {kwargs if kwargs else ''}")

    if image_data_hwc is None:
        _pcb_crop("Erreur: Données image en entrée est None.", lvl="ERROR")
        return None, None
    if wcs_obj is None : #  and ASTROPY_AVAILABLE_IN_UTILS (si wcs_obj peut être autre chose)
        _pcb_crop("Erreur: Objet WCS en entrée est None.", lvl="ERROR")
        return image_data_hwc, None # Retourner l'image, mais pas de WCS

    if not ASTROPY_AVAILABLE_IN_UTILS:
        _pcb_crop("AVERT: Astropy non disponible, impossible d'ajuster le WCS. Rognage de l'image seule effectué si demandé.", lvl="WARN")
        # On pourrait choisir de rogner l'image quand même et retourner un WCS non modifié,
        # ou retourner l'image et WCS originaux. Pour l'instant, on ne touche pas au WCS.
        # Si on rogne l'image, le WCS ne correspondra plus. Il vaut mieux ne pas rogner.
        if crop_percentage_per_side > 1e-4 :
             _pcb_crop(" Rognage annulé car WCS ne peut être ajusté.", lvl="WARN")
        return image_data_hwc, wcs_obj


    if not (0.0 <= crop_percentage_per_side < 0.5):
        if crop_percentage_per_side <= 1e-4 : # Pratiquement pas de rognage demandé
            # _pcb_crop("Pas de rognage demandé (pourcentage nul ou négligeable).", lvl="DEBUG_VERY_DETAIL")
            return image_data_hwc, wcs_obj
        else:
            _pcb_crop(f"Erreur: Pourcentage de rognage ({crop_percentage_per_side*100:.1f}%) hors limites [0, 50).", lvl="ERROR")
            return None, None # Erreur critique si le pourcentage est vraiment invalide


    original_shape = image_data_hwc.shape
    h_orig, w_orig = original_shape[0], original_shape[1]

    dh = int(h_orig * crop_percentage_per_side)
    dw = int(w_orig * crop_percentage_per_side)

    if (2 * dh >= h_orig) or (2 * dw >= w_orig):
        _pcb_crop(f"AVERT: Rognage demandé ({crop_percentage_per_side*100:.1f}%) est trop important pour les dimensions de l'image ({h_orig}x{w_orig}). Rognage annulé.", lvl="WARN")
        return image_data_hwc, wcs_obj

    _pcb_crop(f"Rognage de {dh}px (Haut/Bas) et {dw}px (Gauche/Droite).", lvl="DEBUG_DETAIL")

    if image_data_hwc.ndim == 3: # HWC
        cropped_image_data = image_data_hwc[dh : h_orig - dh, dw : w_orig - dw, :]
    elif image_data_hwc.ndim == 2: # HW
        cropped_image_data = image_data_hwc[dh : h_orig - dh, dw : w_orig - dw]
    else:
        _pcb_crop(f"Erreur: Dimensions d'image non supportées pour le rognage ({image_data_hwc.ndim}D).", lvl="ERROR")
        return None, None # Ou retourner l'original ? Mieux de signaler un échec.
        
    new_h, new_w = cropped_image_data.shape[0], cropped_image_data.shape[1]
    # _pcb_crop(f"Nouvelle shape après rognage: {new_h}x{new_w}", lvl="DEBUG_VERY_DETAIL")

    # Ajuster l'objet WCS
    try:
        # Tenter d'utiliser wcs.slice_like si l'objet WCS le supporte (Astropy >= 5.0)
        # et si l'objet WCS est bien un objet Astropy WCS.
        # Pour cela, il faudrait importer WCS d'Astropy ici.
        # from astropy.wcs import WCS as AstropyWCS (à mettre en haut du fichier utils)
        # if isinstance(wcs_obj, AstropyWCS) and hasattr(wcs_obj, 'slice_like'): # Nécessite l'import
        
        # Pour l'instant, faisons l'ajustement manuel de CRPIX, qui est plus universel
        # mais moins précis pour les WCS complexes.
        
        # Vérifier si wcs_obj est bien un objet WCS d'Astropy avant d'accéder à .wcs
        if not (hasattr(wcs_obj, 'wcs') and hasattr(wcs_obj.wcs, 'crpix')):
            _pcb_crop("AVERT: l'objet WCS ne semble pas être un WCS Astropy standard (manque .wcs.crpix). Ajustement WCS manuel impossible.", lvl="WARN")
            return cropped_image_data, wcs_obj # Retourner l'image rognée, mais le WCS original non modifié

        cropped_wcs_obj = wcs_obj.copy() # Travailler sur une copie

        if cropped_wcs_obj.wcs.crpix is not None:
            # CRPIX est 1-based dans le header FITS, mais l'attribut .wcs.crpix d'un objet WCS Astropy
            # est généralement interprété comme 1-based pour la manipulation via l'API haut niveau.
            # Lorsqu'on soustrait, on soustrait le nombre de pixels rognés du côté "origine" (gauche/bas).
            new_crpix1 = cropped_wcs_obj.wcs.crpix[0] - dw
            new_crpix2 = cropped_wcs_obj.wcs.crpix[1] - dh
            # Clamp to positive values to avoid invalid WCS after heavy cropping
            new_crpix1 = max(new_crpix1, 1.0)
            new_crpix2 = max(new_crpix2, 1.0)
            cropped_wcs_obj.wcs.crpix = [new_crpix1, new_crpix2]
        else:
            # Ce cas est peu probable si c'est un WCS valide, mais par sécurité.
            _pcb_crop("AVERT: wcs_obj.wcs.crpix est None. Impossible d'ajuster CRPIX.", lvl="WARN")
            # On pourrait essayer de retourner wcs_obj original, mais il ne correspondra plus.
            # Renvoyer None pour le WCS est plus sûr pour indiquer un problème.
            return cropped_image_data, None


        # Mettre à jour la taille de l'image de référence dans l'objet WCS
        # pixel_shape est (width, height) et 0-indexed pour l'API Python
        # NAXIS1/2 dans le header sont 1-indexed et (width, height)
        if hasattr(cropped_wcs_obj, 'pixel_shape'):
             cropped_wcs_obj.pixel_shape = (new_w, new_h)
        # Alternativement, si on manipule directement les clés FITS-like dans .wcs:
        if hasattr(cropped_wcs_obj.wcs, 'naxis1'): cropped_wcs_obj.wcs.naxis1 = new_w
        if hasattr(cropped_wcs_obj.wcs, 'naxis2'): cropped_wcs_obj.wcs.naxis2 = new_h
        
        # _pcb_crop("Ajustement WCS terminé.", lvl="DEBUG_DETAIL")
        return cropped_image_data, cropped_wcs_obj

    except Exception as e_wcs_crop:
        _pcb_crop(f"Erreur lors de l'ajustement du WCS: {e_wcs_crop}", lvl="ERROR")
        # Pas de logger global ici, on se fie au progress_callback
        # logger.error(f"Erreur lors de l'ajustement du WCS pour l'image rognée:", exc_info=True)
        return cropped_image_data, None # Retourner l'image rognée mais indiquer échec WCS





def debayer_image(img_norm_01, bayer_pattern="GRBG", progress_callback=None):
    def _log_util_debayer(message, level="DEBUG_DETAIL"):
        if progress_callback and callable(progress_callback): progress_callback(f"  [ZU Debayer] {message}", None, level)
        else: print(f"  [ZU Debayer PRINTFALLBACK] {level}: {message}")
    
    _log_util_debayer(f"Début debayering. Shape entrée: {img_norm_01.shape if hasattr(img_norm_01, 'shape') else 'N/A'}, Pattern: {bayer_pattern}", "DEBUG")
    if not isinstance(img_norm_01, np.ndarray): _log_util_debayer(f"ERREUR: Entrée pas ndarray.", "ERROR"); raise TypeError("Input must be NumPy array")
    if img_norm_01.ndim != 2: _log_util_debayer(f"ERREUR: Attend 2D.", "ERROR"); raise ValueError("Expects 2D image")

    img_uint16 = (np.clip(img_norm_01, 0.0, 1.0) * 65535.0).astype(np.uint16)
    _log_util_debayer(f"Converti en uint16 [0,65535] pour OpenCV.", "DEBUG_DETAIL")
    
    bayer_codes = {"GRBG": cv2.COLOR_BayerGR2RGB, "RGGB": cv2.COLOR_BayerRG2RGB, "GBRG": cv2.COLOR_BayerGB2RGB, "BGGR": cv2.COLOR_BayerBG2RGB}
    bayer_pattern_upper = bayer_pattern.upper()
    if bayer_pattern_upper not in bayer_codes: _log_util_debayer(f"ERREUR: Motif Bayer '{bayer_pattern}' non supporté.", "ERROR"); raise ValueError(f"Bayer pattern '{bayer_pattern}' not supported")
    
    try:
        _log_util_debayer(f"Appel cv2.cvtColor avec code {bayer_codes[bayer_pattern_upper]}...", "DEBUG_DETAIL")
        color_img_bgr_uint16 = cv2.cvtColor(img_uint16, bayer_codes[bayer_pattern_upper])
        color_img_rgb_uint16 = cv2.cvtColor(color_img_bgr_uint16, cv2.COLOR_BGR2RGB)
    except cv2.error as cv_err:
        _log_util_debayer(f"ERREUR OpenCV debayering (pattern: {bayer_pattern_upper}): {cv_err}", "ERROR")
        if progress_callback: progress_callback(f"  [ZU Debayer TRACEBACK] {traceback.format_exc(limit=2)}", None, "ERROR")
        raise ValueError(f"OpenCV error during debayering: {cv_err}")
    
    _log_util_debayer(f"Debayering OK. Conversion retour float32 [0,1]. Shape sortie: {color_img_rgb_uint16.shape}", "DEBUG")
    return color_img_rgb_uint16.astype(np.float32) / 65535.0


def detect_and_correct_hot_pixels(image, threshold=3.0, neighborhood_size=5,
                                  progress_callback=None, save_mask_path=None):
    def _log_util_hp(message, level="DEBUG_DETAIL"):
        if progress_callback and callable(progress_callback): progress_callback(f"  [ZU HotPix] {message}", None, level)
        else: print(f"  [ZU HotPix PRINTFALLBACK] {level}: {message}")

    _log_util_hp(
        f"Début détection/correction HP. Threshold: {threshold}, Neighborhood: {neighborhood_size}",
        "DEBUG",
    )
    if image is None: _log_util_hp("AVERT: Image entrée est None.", "WARN"); return None
    if not isinstance(image, np.ndarray): _log_util_hp(f"ERREUR: Entrée pas ndarray.", "ERROR"); return image 

    if neighborhood_size % 2 == 0: neighborhood_size += 1
    neighborhood_size = max(3, neighborhood_size); ksize = (neighborhood_size, neighborhood_size)

    original_dtype = image.dtype; img_float = image.astype(np.float32, copy=True)
    is_color = img_float.ndim == 3 and img_float.shape[-1] == 3
    _log_util_hp(f"Image {'couleur' if is_color else 'monochrome'}. Dtype original: {original_dtype}.", "DEBUG_DETAIL")
    
    try:
        mask_accum = None
        if is_color:
            mask_accum = np.zeros(img_float.shape, dtype=np.uint8)
            for c in range(img_float.shape[2]):
                channel = img_float[:, :, c]
                median_filtered = cv2.medianBlur(channel, neighborhood_size)
                mean_local = cv2.blur(channel, ksize)
                mean_sq_local = cv2.blur(channel**2, ksize)
                std_dev_local = np.sqrt(np.maximum(mean_sq_local - mean_local**2, 0))
                std_dev_floor = (
                    1e-5
                    if np.issubdtype(channel.dtype, np.floating)
                    else (
                        1.0
                        / (
                            np.iinfo(np.uint16).max
                            if np.max(channel) <= 1
                            else np.iinfo(channel.dtype).max
                            if np.issubdtype(channel.dtype, np.integer)
                            else (2**16 - 1)
                        )
                        if np.max(channel) > 1
                        else 1.0
                    )
                )
                std_dev_local_thresholded = np.maximum(std_dev_local, std_dev_floor)
                hot_pixels_mask = channel > (median_filtered + threshold * std_dev_local_thresholded)
                num_hot = np.sum(hot_pixels_mask)
                if num_hot > 0:
                    _log_util_hp(f"    Canal {c}: {num_hot} pixels chauds corrigés.", "DEBUG_DETAIL")
                channel[hot_pixels_mask] = median_filtered[hot_pixels_mask]
                mask_accum[..., c] = hot_pixels_mask
        else:  # Grayscale
            median_filtered = cv2.medianBlur(img_float, neighborhood_size)
            mean_local = cv2.blur(img_float, ksize)
            mean_sq_local = cv2.blur(img_float**2, ksize)
            std_dev_local = np.sqrt(np.maximum(mean_sq_local - mean_local**2, 0))
            std_dev_floor = (
                1e-5
                if np.issubdtype(img_float.dtype, np.floating)
                else (
                    1.0
                    / (
                        np.iinfo(np.uint16).max
                        if np.max(img_float) <= 1
                        else np.iinfo(img_float.dtype).max
                        if np.issubdtype(img_float.dtype, np.integer)
                        else (2**16 - 1)
                    )
                    if np.max(img_float) > 1
                    else 1.0
                )
            )
            std_dev_local_thresholded = np.maximum(std_dev_local, std_dev_floor)
            hot_pixels_mask = img_float > (median_filtered + threshold * std_dev_local_thresholded)
            num_hot = np.sum(hot_pixels_mask)
            if num_hot > 0:
                _log_util_hp(f"  Image N&B: {num_hot} pixels chauds corrigés.", "DEBUG_DETAIL")
            img_float[hot_pixels_mask] = median_filtered[hot_pixels_mask]
            mask_accum = hot_pixels_mask.astype(np.uint8)
        if save_mask_path:
            try:
                np.save(save_mask_path, mask_accum.astype(np.uint8))
                _log_util_hp(f"Masque HP sauvegardé vers {os.path.basename(save_mask_path)}", "DEBUG_DETAIL")
            except Exception as e_save:
                _log_util_hp(f"ERREUR sauvegarde masque HP: {e_save}", "WARN")
        del mask_accum

        if np.issubdtype(original_dtype, np.integer):
            d_info = np.iinfo(original_dtype)
            corrected_img = np.clip(np.round(img_float), d_info.min, d_info.max).astype(original_dtype)
        else: corrected_img = img_float.astype(original_dtype)
        _log_util_hp(f"Correction HP terminée.", "DEBUG")
        return corrected_img
        
    except cv2.error as cv_err_hp: _log_util_hp(f"ERREUR OpenCV HotPix: {cv_err_hp}", "ERROR"); return image
    except Exception as e_hp: _log_util_hp(f"ERREUR Inattendue HotPix: {e_hp}", "ERROR"); return image


def make_radial_weight_map(height: int, width: int,
                           feather_fraction: float = 0.8,
                           shape_power: float = 2.0,
                           min_weight_floor: float = 0.05, # NOUVEAU PARAMÈTRE, défaut à 0.0 = pas de plancher
                           progress_callback: callable = None) -> np.ndarray: # Ajout de progress_callback pour les logs
    """
    Crée une carte de poids 2D avec une atténuation radiale basée sur une fonction cosinus.
    Le poids est de 1 au centre et décroît jusqu'à 0 (ou min_weight_floor) sur les bords.

    Args:
        height (int): Hauteur de l'image (nombre de lignes).
        width (int): Largeur de l'image (nombre de colonnes).
        feather_fraction (float): Fraction (0.1-1.0) de la demi-diagonale où le poids atteint zéro (ou le plancher).
        shape_power (float): Exposant appliqué à la fonction cosinus (ex: 2.0 pour cos²).
        min_weight_floor (float): Valeur plancher minimale pour les poids (0.0 à <1.0).
                                  Si > 0, les poids ne descendront pas en dessous de cette valeur.
        progress_callback (callable, optional): Fonction pour les logs.

    Returns:
        np.ndarray: Carte de poids 2D de forme (height, width).
    """
    # Alias local pour le callback, si fourni
    _pcb_radial = lambda msg_key, lvl="DEBUG_DETAIL", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else print(f"RADIAL_MAP_LOG {lvl}: {msg_key} {kwargs}")


    if not (0.1 <= feather_fraction <= 1.0):
        _pcb_radial(f"RadialMap: feather_fraction ({feather_fraction}) hors [0.1, 1.0]. Clampe à {np.clip(feather_fraction, 0.1, 1.0)}.", lvl="WARN")
        feather_fraction = np.clip(feather_fraction, 0.1, 1.0)

    if not (0.0 <= min_weight_floor < 1.0):
        _pcb_radial(f"RadialMap: min_weight_floor ({min_weight_floor}) hors [0.0, 1.0). Clampe à {np.clip(min_weight_floor, 0.0, 0.99)}.", lvl="WARN")
        min_weight_floor = np.clip(min_weight_floor, 0.0, 0.99)


    y_coords, x_coords = np.ogrid[:height, :width]
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0

    delta_x = x_coords - center_x
    delta_y = y_coords - center_y
    distance_from_center = np.sqrt(delta_x**2 + delta_y**2)

    max_distance_to_normalize = 0.5 * np.hypot(height, width)
    if max_distance_to_normalize < 1e-6:
        _pcb_radial("RadialMap: Image trop petite, retour poids uniforme 1.0.", lvl="DEBUG_DETAIL")
        return np.ones((height, width), dtype=np.float32)

    normalized_distance = distance_from_center / max_distance_to_normalize
    arg_cos = normalized_distance / feather_fraction
    
    # Calcul de la carte de poids basée sur le cosinus
    weight_map_cos = np.cos(0.5 * np.pi * np.clip(arg_cos, 0.0, 1.0)) ** shape_power
    
    # Application du plancher de poids si spécifié et > 0
    if min_weight_floor > 1e-6: # Utiliser un epsilon pour comparer les floats à zéro
        _pcb_radial(f"RadialMap: Application d'un plancher de poids minimal de {min_weight_floor:.3f}.", lvl="DEBUG_DETAIL")
        final_weight_map = np.maximum(weight_map_cos, min_weight_floor)
    else:
        final_weight_map = weight_map_cos
        
    return final_weight_map.astype(np.float32)

def stretch_auto_asifits_like(img_hwc_adu, p_low=0.5, p_high=99.8, 
                              asinh_a=0.01, apply_wb=True):
    """
    Étirement type ASIFitsViewer avec asinh et auto balance RVB.
    Fallback vers du linéaire si dynamique trop faible.
    """
    img = img_hwc_adu.astype(np.float32, copy=False)
    out = np.empty_like(img)

    for c in range(3):
        chan = img[..., c]
        vmin, vmax = np.percentile(chan, [p_low, p_high])
        if vmax - vmin < 1e-3:
            out[..., c] = np.zeros_like(chan)
            continue
        normed = np.clip((chan - vmin) / (vmax - vmin), 0, 1)
        # stretch asinh
        stretched = np.arcsinh(normed / asinh_a) / np.arcsinh(1 / asinh_a)
        if np.nanmax(stretched) < 0.05:  # cas trop sombre
            stretched = normed  # fallback linéaire
        out[..., c] = stretched

    if apply_wb:
        avg_per_chan = np.mean(out, axis=(0, 1))
        norm = np.max(avg_per_chan)
        if norm > 0:
            avg_per_chan /= norm
        else:
            avg_per_chan = np.ones_like(avg_per_chan)
        for c in range(3):
            denom = avg_per_chan[c]
            if denom > 1e-8:
                out[..., c] /= denom

    return np.clip(out, 0, 1)

def stretch_percentile_rgb(img_hwc_adu, p_low=0.5, p_high=99.8, 
                           independent_channels=False, 
                           asinh_a=0.05, # 'a' parameter for AsinhStretch
                           progress_callback: callable = None):
    """
    Applique un stretch par percentiles avec une transformation asinh (via Astropy)
    à une image HWC. Sortie normalisée [0,1] pour affichage.

    Args:
        img_hwc_adu (np.ndarray): Image HWC (ou HW), float32, en ADU.
        p_low (float): Percentile inférieur pour définir le point noir (0-100).
        p_high (float): Percentile supérieur pour définir le point blanc (0-100).
        independent_channels (bool): Si True et image couleur, stretch chaque canal indépendamment.
                                     Si False, calcule les limites sur la luminance et applique
                                     le même vmin/vmax à chaque canal avant leur stretch individuel.
        asinh_a (float): Paramètre 'a' pour AsinhStretch. Contrôle la linéarité pour les faibles
                         signaux. Typiquement entre 0.01 (fort) et 1.0 (doux). 0.1 est un bon début.
        progress_callback (callable, optional): Fonction pour les logs.

    Returns:
        np.ndarray: Image HWC (ou HW) normalisée [0,1] après étirement asinh, float32.
                    Retourne une version basique étirée si Astropy.visualization n'est pas disponible.
    """
    if img_hwc_adu is None:
        if progress_callback:
            progress_callback("stretch_utils_error_input_none", lvl="ERROR")
        return None

    if not ASTROPY_VISUALIZATION_AVAILABLE:
        if progress_callback:
            progress_callback("stretch_utils_warn_astropy_viz_unavailable_fallback_linear", lvl="WARN")
        # Fallback très basique si astropy.visualization n'est pas là :
        try:
            # Utiliser les p_low/p_high comme pourcentages
            min_val, max_val = np.percentile(img_hwc_adu, [p_low, p_high])
            if not (np.isfinite(min_val) and np.isfinite(max_val)) or (max_val - min_val < 1e-5):
                return np.zeros_like(img_hwc_adu, dtype=np.float32)
            return np.clip((img_hwc_adu - min_val) / (max_val - min_val), 0, 1).astype(np.float32)
        except Exception as e_fallback:
            if progress_callback:
                progress_callback(f"stretch_utils_error_fallback_stretch: {e_fallback}", lvl="ERROR")
            return np.zeros_like(img_hwc_adu, dtype=np.float32) # Sécurité ultime

    
    img_float = img_hwc_adu.astype(np.float32, copy=False) 

    stretch = AsinhStretch(a=asinh_a)
    
    if img_float.ndim == 2: # Image monochrome
        # CORRECTION : Ajout de n_samples=None
        interval = PercentileInterval(p_low, p_high, n_samples=None)
        try:
            norm = ImageNormalize(img_float, interval=interval, stretch=stretch, clip=True)
            return norm(img_float).astype(np.float32)
        except Exception as e_norm_mono:
            if progress_callback:
                progress_callback(f"stretch_utils_error_norm_mono: {e_norm_mono}", lvl="ERROR")
            return np.zeros_like(img_float, dtype=np.float32) # Fallback
        
    elif img_float.ndim == 3 and img_float.shape[2] == 3: # Image couleur HWC
        stretched_img_array = np.empty_like(img_float)
        if independent_channels:
            for c in range(3):
                # CORRECTION : Ajout de n_samples=None
                interval = PercentileInterval(p_low, p_high, n_samples=None)
                try:
                    norm = ImageNormalize(img_float[..., c], interval=interval, stretch=stretch, clip=True)
                    stretched_img_array[..., c] = norm(img_float[..., c])
                except Exception as e_norm_color_ind:
                    if progress_callback:
                        progress_callback(f"stretch_utils_error_norm_color_ind_ch{c}: {e_norm_color_ind}", lvl="ERROR")
                    stretched_img_array[..., c] = np.zeros_like(img_float[..., c], dtype=np.float32) # Fallback pour ce canal
        else: # Stretch lié basé sur la luminance pour vmin/vmax, mais stretch asinh par canal
            try:
                luminance = 0.299 * img_float[..., 0] + 0.587 * img_float[..., 1] + 0.114 * img_float[..., 2]
                # CORRECTION : Ajout de n_samples=None
                interval = PercentileInterval(p_low, p_high, n_samples=None)
                vmin, vmax = interval.get_limits(luminance)
                
                if not (np.isfinite(vmin) and np.isfinite(vmax)) or (vmax - vmin < 1e-5) : # Vérifier si les limites sont valides
                    if progress_callback:
                        progress_callback(f"stretch_utils_warn_invalid_lum_limits_linked_stretch: vmin={vmin}, vmax={vmax}. Fallback sur stretch indépendant ou neutre.", lvl="WARN")
                    # Fallback : si les limites de luminance sont mauvaises, on pourrait faire un stretch indépendant prudent.
                    # Pour simplifier, on peut retourner une image neutre ou logguer une erreur plus sévère.
                    # Ici, on va essayer un stretch indépendant comme fallback.
                    for c_fb in range(3):
                        interval_fb = PercentileInterval(p_low, p_high, n_samples=None)
                        norm_fb = ImageNormalize(img_float[..., c_fb], interval=interval_fb, stretch=stretch, clip=True)
                        stretched_img_array[..., c_fb] = norm_fb(img_float[..., c_fb])
                    return stretched_img_array.astype(np.float32)

                for c in range(3):
                    norm = ImageNormalize(img_float[..., c], vmin=vmin, vmax=vmax, stretch=stretch, clip=True)
                    stretched_img_array[..., c] = norm(img_float[..., c])
            except Exception as e_norm_color_linked:
                if progress_callback:
                    progress_callback(f"stretch_utils_error_norm_color_linked: {e_norm_color_linked}", lvl="ERROR")
                # Fallback en cas d'erreur majeure dans le stretch lié
                for c_fb_err in range(3): stretched_img_array[..., c_fb_err] = np.clip(img_float[...,c_fb_err] / (np.max(img_float[...,c_fb_err]) if np.max(img_float[...,c_fb_err]) > 0 else 1.0) ,0,1)


        return stretched_img_array.astype(np.float32)
    else:
        if progress_callback:
            progress_callback("stretch_utils_warn_unsupported_shape_for_stretch", lvl="WARN", shape=str(img_float.shape if hasattr(img_float, 'shape') else 'N/A'))
        return img_float.astype(np.float32) # Retourner en float32 au cas où


def save_numpy_to_fits(image_data: np.ndarray, header, output_path: str, *, axis_order: str = "HWC", overwrite: bool = True) -> None:
    """Write a NumPy array to FITS without any scaling.

    Parameters
    ----------
    image_data : np.ndarray
        Array to save. Can be 2-D or 3-D.
    header : fits.Header or dict
        Header to write with the data.
    output_path : str
        Destination FITS path.
    axis_order : {"HWC", "CHW"}
        Interpretation of 3-D arrays. ``HWC`` means channels last and the array
        will be transposed to ``CxHxW`` for saving.
    overwrite : bool
        Overwrite existing file if True.
    """

    current_fits = fits_module_for_utils
    final_header = current_fits.Header()
    if header is not None:
        try:
            if hasattr(header, "to_header"):
                final_header.update(header.to_header(relax=True))
            else:
                final_header.update(header)
        except Exception:
            pass

    for key in ["SIMPLE", "BITPIX", "NAXIS", "EXTEND", "BSCALE", "BZERO"]:
        if key in final_header:
            try:
                del final_header[key]
            except KeyError:
                pass

    data_to_write = image_data
    if image_data.ndim == 3:
        ao = str(axis_order).upper()
        if ao == "HWC":
            data_to_write = np.moveaxis(image_data, -1, 0)
    hdu = current_fits.PrimaryHDU(data=data_to_write, header=final_header)
    hdul = current_fits.HDUList([hdu])
    hdul.writeto(output_path, overwrite=overwrite)
    if hasattr(hdul, "close"):
        hdul.close()




def save_fits_image(image_data: np.ndarray,
                    output_path: str,
                    header = None,  # Type hint peut être plus flexible: fits_module_for_utils.Header() | dict
                    overwrite: bool = True,
                    save_as_float: bool = False,
                    progress_callback: callable = None,
                    axis_order: str = "HWC"):
    """
    Sauvegarde des données image NumPy dans un fichier FITS.
    Utilise ASTROPY_AVAILABLE_IN_UTILS défini localement.
    Version avec logs de débogage améliorés et gestion gc.
    L'argument ``axis_order`` indique comment interpréter les tableaux couleur
    en entrée.
    - ``"HWC"`` (défaut) : ``Height x Width x Channels``. Les données sont
      transposées en ``CxHxW`` pour l'écriture FITS.
    - ``"CHW"`` : les données sont déjà dans l'ordre ``Channels x Height x Width``.
    """

    def _log_util_save(message, level="DEBUG_DETAIL", pcb=progress_callback):
        log_prefix = "  [ZU SaveFITS]"
        if "SAVE_DEBUG" in message: log_prefix = "  [ZU SaveFITS DEBUG]"
        full_message = f"{log_prefix} {message}"
        if pcb and callable(pcb): pcb(full_message, None, level)
        else: print(full_message)

    base_output_filename = os.path.basename(output_path)
    _log_util_save(f"Début sauvegarde FITS vers '{base_output_filename}'. SaveAsFloat={save_as_float}", "INFO")

    # Utiliser le fits_module_for_utils défini globalement dans ce module
    current_fits_module = fits_module_for_utils 
    current_astropy_available_flag = ASTROPY_AVAILABLE_IN_UTILS

    if not current_astropy_available_flag and current_fits_module.__class__.__name__ == "MockFitsModule":
        _log_util_save(f"ERREUR CRITIQUE: Astropy non disponible. Sauvegarde réelle de '{base_output_filename}' impossible (mock actif).", "ERROR")
        return

    if image_data is None: _log_util_save(f"ERREUR: Image data est None pour '{base_output_filename}'. Sauvegarde annulée.", "ERROR"); return
    if not isinstance(image_data, np.ndarray): _log_util_save(f"ERREUR: Input doit être NumPy array, reçu {type(image_data)}.", "ERROR"); return

    _log_util_save(f"SAVE_DEBUG: Données image_data reçues - Shape: {image_data.shape}, Dtype: {image_data.dtype}, Range: [{np.nanmin(image_data):.3g} - {np.nanmax(image_data):.3g}], IsFinite: {np.all(np.isfinite(image_data))}", "WARN")

    final_header_to_write = current_fits_module.Header()
    if header is not None:
        try:
            if hasattr(header, 'to_header') and callable(header.to_header): final_header_to_write.update(header.to_header(relax=True))
            elif isinstance(header, (current_fits_module.Header if current_astropy_available_flag else dict)): final_header_to_write.update(header.copy()) # type: ignore
            elif isinstance(header, dict): final_header_to_write.update(header)
            else: _log_util_save(f"AVERT: Type de header non supporté ({type(header)}). Utilisation header vide.", "WARN")
        except Exception as e_hdr_copy:
             _log_util_save(f"AVERT: Erreur copie/update header: {e_hdr_copy}. Header partiel/vide.", "WARN")
             final_header_to_write = current_fits_module.Header()

    keywords_to_remove_base = ['SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'BSCALE', 'BZERO']
    for key_k in keywords_to_remove_base:
        if key_k in final_header_to_write:
            try: del final_header_to_write[key_k]
            except KeyError: pass

    data_to_write_temp = None
    if save_as_float:
        data_to_write_temp = image_data.astype(np.float32)
        final_header_to_write['BITPIX'] = -32
        final_header_to_write['BSCALE'] = 1.0
        final_header_to_write['BZERO'] = 0.0
        _log_util_save(f"  SAVE_DEBUG: (Float) data_to_write_temp: Range [{np.nanmin(data_to_write_temp):.3g}, {np.nanmax(data_to_write_temp):.3g}], IsFinite: {np.all(np.isfinite(data_to_write_temp))}", "WARN")
    else:
        min_in, max_in = np.nanmin(image_data), np.nanmax(image_data)
        image_normalized_01 = np.zeros_like(image_data, dtype=np.float32)
        if np.isfinite(min_in) and np.isfinite(max_in) and (max_in > min_in + 1e-9):
            image_normalized_01 = (image_data.astype(np.float32) - min_in) / (max_in - min_in)
        elif np.any(np.isfinite(image_data)): image_normalized_01 = np.full_like(image_data, 0.5, dtype=np.float32)
        
        image_clipped_01 = np.clip(image_normalized_01, 0.0, 1.0)
        data_to_write_temp = (image_clipped_01 * 65535.0).astype(np.uint16)
        final_header_to_write['BITPIX'] = 16; final_header_to_write['BSCALE'] = 1; final_header_to_write['BZERO'] = 32768
        _log_util_save(f"  SAVE_DEBUG: (Uint16) data_to_write_temp: Range [{np.min(data_to_write_temp)}, {np.max(data_to_write_temp)}]", "WARN")

    data_for_hdu_cxhxw = None
    is_color = data_to_write_temp.ndim == 3 and data_to_write_temp.shape[-1] == 3
    if is_color:
        axis_order_upper = str(axis_order).upper()
        if axis_order_upper == 'HWC':
            h, w, c = data_to_write_temp.shape
            data_for_hdu_cxhxw = np.moveaxis(data_to_write_temp, -1, 0)
        elif axis_order_upper == 'CHW':
            c, h, w = data_to_write_temp.shape
            data_for_hdu_cxhxw = data_to_write_temp
        else:
            _log_util_save(f"Axis order '{axis_order}' non reconnu, utilisation 'HWC'", "WARN")
            h, w, c = data_to_write_temp.shape
            data_for_hdu_cxhxw = np.moveaxis(data_to_write_temp, -1, 0)
        final_header_to_write['NAXIS'] = 3
        final_header_to_write['NAXIS1'] = w
        final_header_to_write['NAXIS2'] = h
        final_header_to_write['NAXIS3'] = c
        if 'CTYPE3' not in final_header_to_write:
            final_header_to_write['CTYPE3'] = ('RGB', 'Color Format')
        if 'EXTNAME' not in final_header_to_write:
            final_header_to_write['EXTNAME'] = 'RGB'
    else: # HW (ou déjà CxHxW, par exemple une carte de couverture)
        if data_to_write_temp.ndim == 2: # Cas explicite HW pour monochrome
            data_for_hdu_cxhxw = data_to_write_temp
            final_header_to_write['NAXIS'] = 2; final_header_to_write['NAXIS1'] = data_to_write_temp.shape[1]
            final_header_to_write['NAXIS2'] = data_to_write_temp.shape[0]
            if 'NAXIS3' in final_header_to_write: del final_header_to_write['NAXIS3']
            if 'CTYPE3' in final_header_to_write: del final_header_to_write['CTYPE3']
        else: # Si ce n'est ni HWC ni HW, on suppose que c'est déjà dans le bon format pour HDU (ex: couverture CxHxW)
            data_for_hdu_cxhxw = data_to_write_temp 
            # Dans ce cas, on espère que le header contient déjà les bonnes infos NAXIS, ou Astropy les déduira.
            _log_util_save(f"SAVE_DEBUG: Shape data_to_write_temp non HWC ni HW standard: {data_to_write_temp.shape}. Passage direct à HDU.", "DEBUG_DETAIL")

    
    _log_util_save(f"SAVE_DEBUG: Données PRÊTES pour HDU (data_for_hdu_cxhxw) - Shape: {data_for_hdu_cxhxw.shape}, Dtype: {data_for_hdu_cxhxw.dtype}, Range: [{np.nanmin(data_for_hdu_cxhxw):.3g}, {np.nanmax(data_for_hdu_cxhxw):.3g}], IsFinite: {np.all(np.isfinite(data_for_hdu_cxhxw))}", "WARN")
        
    hdul = None 
    primary_hdu_object = None # Pour pouvoir del primary_hdu.data
    try:
        _log_util_save(f"SAVE_DEBUG: AVANT PrimaryHDU - data_for_hdu_cxhxw - Min: {np.nanmin(data_for_hdu_cxhxw)}, Max: {np.nanmax(data_for_hdu_cxhxw)}, Mean: {np.nanmean(data_for_hdu_cxhxw)}, Std: {np.nanstd(data_for_hdu_cxhxw)}, Dtype: {data_for_hdu_cxhxw.dtype}, Finite: {np.all(np.isfinite(data_for_hdu_cxhxw))}", "ERROR")
        
        primary_hdu_object = current_fits_module.PrimaryHDU(data=data_for_hdu_cxhxw, header=final_header_to_write)
        
        _log_util_save(f"SAVE_DEBUG: APRÈS PrimaryHDU - primary_hdu.data - Min: {np.nanmin(primary_hdu_object.data)}, Max: {np.nanmax(primary_hdu_object.data)}, Mean: {np.nanmean(primary_hdu_object.data)}, Dtype: {primary_hdu_object.data.dtype}, Finite: {np.all(np.isfinite(primary_hdu_object.data))}", "ERROR")
        
        hdul = current_fits_module.HDUList([primary_hdu_object])
        _log_util_save(f"Écriture vers '{base_output_filename}' (overwrite={overwrite})...", "DEBUG_DETAIL")
        
        hdul.writeto(output_path, overwrite=overwrite, checksum=True, output_verify='exception') 
        _log_util_save(f"Sauvegarde FITS vers '{base_output_filename}' RÉUSSIE.", "INFO")

    except Exception as e_write:
        _log_util_save(f"ERREUR CRITIQUE lors sauvegarde FITS '{base_output_filename}': {type(e_write).__name__} - {e_write}", "ERROR")
        if progress_callback: _log_util_save(f"  [ZU SaveFITS TRACEBACK] {traceback.format_exc(limit=3)}", "ERROR")
    finally:
        if hdul is not None and hasattr(hdul, 'close'):
            try: hdul.close()
            except Exception: pass
        
        # Nettoyage explicite pour aider le GC
        if 'data_to_write_temp' in locals() and data_to_write_temp is not None:
            del data_to_write_temp
        if 'data_for_hdu_cxhxw' in locals() and data_for_hdu_cxhxw is not None:
            del data_for_hdu_cxhxw
        if primary_hdu_object is not None and hasattr(primary_hdu_object, 'data') and primary_hdu_object.data is not None:
             del primary_hdu_object.data 
        if primary_hdu_object is not None:
             del primary_hdu_object
        if 'hdul' in locals() and hdul is not None:
            del hdul
        gc.collect() # gc doit être importé en haut du fichier zemosaic_utils.py


def gpu_assemble_final_mosaic_reproject_coadd(*args, **kwargs):
    """GPU accelerated final mosaic assembly (reproject & coadd).

    This is a placeholder implementation. A full version would mirror the
    NumPy implementation using CuPy arrays and CUDA kernels while minimizing
    data transfers between host and device.
    """
    raise NotImplementedError("GPU implementation not available")


def gpu_assemble_final_mosaic_incremental(*args, **kwargs):
    """GPU accelerated incremental mosaic assembly placeholder."""
    raise NotImplementedError("GPU implementation not available")


def gpu_reproject_and_coadd(data_list, wcs_list, shape_out, **kwargs):
    """Simplified GPU implementation using CuPy."""
    import cupy as cp
    data_gpu = [cp.asarray(d) for d in data_list]
    mosaic_gpu = cp.zeros(shape_out, dtype=cp.float32)
    weight_gpu = cp.zeros(shape_out, dtype=cp.float32)
    for img in data_gpu:
        # Placeholder for GPU interpolation logic
        pass
    return cp.asnumpy(mosaic_gpu), cp.asnumpy(weight_gpu)


def reproject_and_coadd_wrapper(
    data_list,
    wcs_list,
    shape_out,
    use_gpu=False,
    cpu_function=None,
    **kwargs,
):
    if use_gpu and GPU_AVAILABLE:
        try:
            return gpu_reproject_and_coadd(data_list, wcs_list, shape_out, **kwargs)
        except Exception as e:  # pragma: no cover - GPU failures
            import logging

            logging.getLogger(__name__).warning(
                "GPU reprojection failed (%s), fallback CPU", e
            )
    if cpu_function is None:
        cpu_function = cpu_reproject_and_coadd
    inputs = list(zip(data_list, wcs_list))
    output_proj = kwargs.pop("output_projection")
    return cpu_function(inputs, output_proj, shape_out, **kwargs)



def gpu_reproject_and_coadd(data_list, wcs_list, shape_out, **kwargs):
    """Simplified GPU version of ``reproject_and_coadd``.

    Parameters match :func:`reproject_and_coadd_wrapper` but operate on CuPy
    arrays. The implementation here is schematic and should be replaced with a
    real CUDA accelerated routine.
    """
    import cupy as cp
    data_gpu = [cp.asarray(d) for d in data_list]
    mosaic_gpu = cp.zeros(shape_out, dtype=cp.float32)
    weight_gpu = cp.zeros(shape_out, dtype=cp.float32)
    for img in data_gpu:
        # Placeholder for GPU interpolation step
        pass
    return cp.asnumpy(mosaic_gpu), cp.asnumpy(weight_gpu)


def reproject_and_coadd_wrapper(data_list, wcs_list, shape_out, use_gpu=False, cpu_func=None, **kwargs):
    """Dispatch to CPU or GPU ``reproject_and_coadd`` depending on availability."""
    if use_gpu and GPU_AVAILABLE:
        try:
            return gpu_reproject_and_coadd(data_list, wcs_list, shape_out, **kwargs)
        except Exception as e:  # pragma: no cover - GPU errors
            import logging
            logging.getLogger(__name__).warning(
                "GPU reprojection failed (%s), fallback CPU", e
            )
    input_pairs = list(zip(data_list, wcs_list))
    output_projection = kwargs.pop("output_projection", None)
    func = cpu_func or cpu_reproject_and_coadd
    return func(input_pairs, output_projection, shape_out, **kwargs)






#####################################################################################################################

