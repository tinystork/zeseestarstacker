# --- START OF FILE seestar/core/image_processing.py ---
"""
Fonctions de base pour le traitement d'images astronomiques.
"""
import os
import numpy as np
from astropy.io import fits
import cv2
import warnings
from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyWarning 
from PIL import Image, ImageEnhance, ImageFilter # Added PIL imports for enhanced stretch
from astropy.stats import sigma_clipped_stats
import traceback # Pour un meilleur débogage des erreurs de lecture FITS
from astropy.io import fits
import sys
warnings.filterwarnings("ignore", category=FutureWarning)



def load_and_validate_fits(filepath, normalize_to_float32=True, attempt_fix_nonfinite=True):
    """
    Charge une image FITS, la valide, la normalise en float32 [0,1] et gère la transposition.
    Version: V2.1 (Gère CxHxW et HxWxC, logs améliorés, fallback header)

    Args:
        filepath (str): Chemin vers le fichier FITS.
        normalize_to_float32 (bool): Si True, normalise l'image en float32 [0,1].
        attempt_fix_nonfinite (bool): Si True, tente de remplacer NaN/Inf par 0.

    Returns:
        tuple: (image_data, header) ou (None, header_fallback) en cas d'échec.
               image_data est np.ndarray (HxW ou HxWxC) float32 [0,1] si normalisé,
               sinon les données brutes.
               header est l'objet astropy.io.fits.Header.
               header_fallback est un header (potentiellement partiel ou vide) si la lecture des données échoue.
    """
    filename = os.path.basename(filepath)
    print(f"DEBUG IP (load_and_validate_fits V2.1): Début chargement pour '{filename}'")
    
    data_raw = None
    header = None
    header_for_fallback = fits.Header() # Header vide par défaut pour fallback
    img_hdu_idx = -1 # Pour loguer quelle HDU a été utilisée

    try:
        with fits.open(filepath, memmap=False, do_not_scale_image_data=True) as hdul:
            if not hdul:
                print(f"  REJET (load_and_validate_fits V2.1): Fichier FITS vide ou corrompu: '{filename}'")
                return None, header_for_fallback

            # Essayer de trouver la première HDU image valide
            hdu_img = None
            for idx, hdu_item in enumerate(hdul):
                if hdu_item.is_image and hasattr(hdu_item, 'data') and hdu_item.data is not None:
                    # Si c'est une HDU principale ou une extension nommée 'SCI' ou 'IMAGE'
                    if idx == 0 or \
                       (hasattr(hdu_item, 'name') and isinstance(hdu_item.name, str) and 
                        hdu_item.name.upper() in ['SCI', 'IMAGE', 'PRIMARY']):
                        hdu_img = hdu_item
                        img_hdu_idx = idx
                        break # Prendre la première de ce type
            
            # Si pas trouvée avec nom/index prioritaire, prendre la première HDU image non vide
            if hdu_img is None:
                for idx, hdu_item in enumerate(hdul):
                    if hdu_item.is_image and hasattr(hdu_item, 'data') and hdu_item.data is not None:
                        hdu_img = hdu_item
                        img_hdu_idx = idx
                        break
            
            if hdu_img is None:
                print(f"  REJET (load_and_validate_fits V2.1): Aucune HDU image valide trouvée dans '{filename}'.")
                # Essayer de récupérer au moins le header primaire s'il existe
                if len(hdul) > 0 and hdul[0].header:
                    header_for_fallback = hdul[0].header.copy()
                return None, header_for_fallback

            data_raw = hdu_img.data
            header = hdu_img.header.copy() # Toujours prendre une copie
            header_for_fallback = header.copy() # Mettre à jour le fallback avec le header trouvé
            
            print(f"  DEBUG IP (load_and_validate_fits V2.1): Données image trouvées dans HDU {img_hdu_idx} pour '{filename}'. Shape brute: {data_raw.shape if data_raw is not None else 'None'}, Dtype brut: {data_raw.dtype if data_raw is not None else 'None'}")

            if data_raw is None:
                raise ValueError("Données image (data_raw) sont None après lecture HDU.")
            
            print(f"  DEBUG IP (load_and_validate_fits V2.1): Validation données brutes pour '{filename}'. Shape: {data_raw.shape}, Dtype: {data_raw.dtype}")

            # --- VALIDATION ET TRANSPOSITION SI NÉCESSAIRE ---
            if data_raw.ndim == 3:
                # Cas 1: CxHxW (souvent 3xHxW pour RGB ou N fichiers empilés)
                # Heuristique : si le premier axe est petit (1,3,4) et les autres grands
                if data_raw.shape[0] in [1, 3, 4] and data_raw.shape[1] > 4 and data_raw.shape[2] > 4:
                    print(f"    INFO IP: Détection format CxHxW ({data_raw.shape}). Transposition en HxWxC...")
                    data_raw = np.moveaxis(data_raw, 0, -1) 
                    print(f"    INFO IP: Shape après transposition: {data_raw.shape}")
                # Cas 2: HxWxC (déjà au bon format)
                elif data_raw.shape[2] in [1, 3, 4] and data_raw.shape[0] > 4 and data_raw.shape[1] > 4:
                    print(f"    INFO IP: Détection format HxWxC ({data_raw.shape}). Aucune transposition nécessaire.")
                    pass 
                else:
                    print(f"  REJET (load_and_validate_fits V2.1): Shape de données brutes 3D non supportée ({data_raw.shape}) pour '{filename}'. Les axes doivent être clairement (C,H,W) ou (H,W,C) avec C=1,3,4.")
                    return None, header # Retourner le header lu même si les données sont rejetées
            elif data_raw.ndim != 2:
                print(f"  REJET (load_and_validate_fits V2.1): Shape de données {data_raw.ndim}D non supportée pour '{filename}'. Doit être 2D ou 3D (HxW, HxWxC, ou CxHxW).")
                return None, header # Retourner le header lu
            # Si data_raw.ndim == 2, c'est un format N&B HxW, ce qui est valide.
            
            # --- FIN VALIDATION ET TRANSPOSITION ---

            # Gérer les valeurs non finies si demandé
            if attempt_fix_nonfinite and not np.all(np.isfinite(data_raw)):
                print(f"    WARN IP: Données non finies détectées dans '{filename}'. Remplacement par 0.")
                data_raw = np.nan_to_num(data_raw, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalisation optionnelle
            if normalize_to_float32:
                print(f"  DEBUG IP (load_and_validate_fits V2.1): Préparation pour normalisation de '{filename}'...")
                # Convertir en float64 pour les stats pour éviter overflow/underflow avec entiers
                try:
                    data_stats_type = data_raw.astype(np.float64) 
                    print(f"    DEBUG IP: Conversion data_raw (dtype {data_raw.dtype}) vers float64 pour stats...")
                except Exception as e_astype_stats:
                    print(f"    WARN IP: Échec conversion pour stats float64: {e_astype_stats}. Utilisation type original.")
                    data_stats_type = data_raw # Fallback
                
                min_val, max_val = np.nanmin(data_stats_type), np.nanmax(data_stats_type)
                print(f"    DEBUG IP: Stats sur {data_stats_type.dtype} pour '{filename}': Min={min_val}, Max={max_val} (dtype_stats: {data_stats_type.dtype})")

                # Convertir en float32 pour la normalisation et la sortie
                data_float32_to_norm = data_raw.astype(np.float32)
                print(f"    DEBUG IP: data_float32_to_norm (pour normalisation) - dtype: {data_float32_to_norm.dtype}, shape: {data_float32_to_norm.shape}")

                if np.isfinite(min_val) and np.isfinite(max_val) and (max_val > min_val):
                    denominator = max_val - min_val
                    print(f"    DEBUG IP: Normalisation standard pour '{filename}' (denominator={denominator:.3g})...")
                    image_data_normalized = (data_float32_to_norm - min_val) / denominator
                    image_data = np.clip(image_data_normalized, 0.0, 1.0).astype(np.float32)
                elif np.any(np.isfinite(data_float32_to_norm)): # Image constante non-Nan/Inf
                    print(f"    WARN IP: Image '{filename}' semble constante (min={min_val}, max={max_val}). Normalisation à 0.5.")
                    image_data = np.full_like(data_float32_to_norm, 0.5, dtype=np.float32)
                else: # Toutes les valeurs sont NaN ou Inf
                    print(f"    WARN IP: Image '{filename}' ne contient que des valeurs non finies. Normalisation à 0.0.")
                    image_data = np.zeros_like(data_float32_to_norm, dtype=np.float32)
            else: # Pas de normalisation, retourner les données brutes (potentiellement transposées)
                image_data = data_raw.astype(np.float32) # Assurer float32 en sortie quand même

            mean_val = np.nanmean(image_data)
            std_val = np.nanstd(image_data)
            print(f"  DEBUG IP (load_and_validate_fits V2.1): FIN pour '{filename}'. Shape sortie: {image_data.shape}, Dtype: {image_data.dtype}. Range final: [{np.nanmin(image_data):.4f} - {np.nanmax(image_data):.4f}], Moyenne: {mean_val:.4f}, StdDev: {std_val:.6f}")
            return image_data, header

    except FileNotFoundError:
        print(f"  ERREUR IP (load_and_validate_fits V2.1): Fichier non trouvé: '{filepath}'")
        return None, header_for_fallback # Retourner un header vide si le fichier n'est pas trouvé
    except MemoryError as me:
        print(f"  ERREUR IP (load_and_validate_fits V2.1): ERREUR MÉMOIRE lors du chargement de '{filename}': {me}")
        traceback.print_exc(limit=1)
        return None, header_for_fallback
    except Exception as e:
        print(f"  ERREUR IP (load_and_validate_fits V2.1): Erreur inattendue lors du chargement/validation de '{filename}': {e}")
        traceback.print_exc(limit=2) # Afficher plus de détails pour les erreurs inattendues
        return None, header_for_fallback

def debayer_image(img, bayer_pattern="GRBG"):
    """
    Convertit une image brute Bayer (normalisée 0-1 float32) en image RGB (0-1 float32).

    Parameters:
        img (numpy.ndarray): Image brute (2D, float32, 0-1).
        bayer_pattern (str): Motif Bayer ("GRBG", "RGGB", "GBRG", "BGGR").

    Returns:
        numpy.ndarray: Image RGB débayerisée (float32, 0-1).

    Raises:
        ValueError: Si le motif Bayer n'est pas supporté ou l'image n'est pas 2D.
    """
    if img.ndim != 2:
        raise ValueError(f"Le debayering ne s'applique qu'aux images 2D. Shape reçue: {img.shape}")

    # Scale to uint16 for OpenCV debayering functions
    img_uint16 = (np.clip(img, 0.0, 1.0) * 65535.0).astype(np.uint16)

    # Mapping from Bayer pattern string to OpenCV code
    bayer_codes = {
        "GRBG": cv2.COLOR_BayerGR2RGB,
        "RGGB": cv2.COLOR_BayerRG2RGB,
        "GBRG": cv2.COLOR_BayerGB2RGB,
        "BGGR": cv2.COLOR_BayerBG2RGB,
    }

    bayer_pattern_upper = bayer_pattern.upper()
    if bayer_pattern_upper in bayer_codes:
        cv_code = bayer_codes[bayer_pattern_upper]
        try:
             # Note: OpenCV Bayer functions output BGR format
             color_img_bgr_uint16 = cv2.cvtColor(img_uint16, cv_code)
             # Convert BGR to RGB
             color_img_rgb_uint16 = cv2.cvtColor(color_img_bgr_uint16, cv2.COLOR_BGR2RGB)
        except cv2.error as cv_err:
             raise ValueError(f"OpenCV error during debayering ({bayer_pattern_upper}): {cv_err}")
    else:
        raise ValueError(f"Motif Bayer '{bayer_pattern}' non supporté. Options: GRBG, RGGB, GBRG, BGGR.")

    # Convert back to float32 (0-1 range)
    return color_img_rgb_uint16.astype(np.float32) / 65535.0


def save_fits_image(image, output_path, header=None, overwrite=True):
    """
    Enregistre une image (normalisée 0-1 float32) au format FITS (uint16).
    Suppresses specific FITS standard VerifyWarnings about keyword length.

    Parameters:
        image (numpy.ndarray): Image 2D (HxW) ou 3D (HxWx3) à enregistrer (float32, 0-1).
        output_path (str): Chemin du fichier de sortie.
        header (astropy.io.fits.Header, optional): En-tête FITS.
        overwrite (bool): Écrase le fichier existant si True.
    """
    # --- (Input validation remains the same) ---
    if image is None:
        print(f"Error: Cannot save None image to {output_path}")
        return
    if not isinstance(image, np.ndarray):
        print(f"Error: Input for save_fits_image must be a numpy array, got {type(image)}")
        return

    # --- (Header creation/copying logic remains the same) ---
    final_header = fits.Header()
    if header is not None and isinstance(header, fits.Header):
         keywords_to_remove = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3',
                              'EXTEND', 'BSCALE', 'BZERO']
         temp_header = header.copy()
         for key in keywords_to_remove:
              if key in temp_header: del temp_header[key]
         final_header.update(temp_header)
    elif header is not None:
         print("Warning: Provided header is not valid astropy.io.fits.Header. Creating new one.")


    # --- (Data preparation logic remains the same) ---
    is_color = image.ndim == 3 and image.shape[-1] == 3
    image_float32 = image.astype(np.float32)
    image_clipped = np.clip(image_float32, 0.0, 1.0)
    image_uint16 = (image_clipped * 65535.0).astype(np.uint16)

    if is_color:
        image_to_save = np.moveaxis(image_uint16, -1, 0)
        final_header['NAXIS'] = 3; final_header['NAXIS1'] = image.shape[1]
        final_header['NAXIS2'] = image.shape[0]; final_header['NAXIS3'] = 3
        if 'CTYPE3' not in final_header: final_header['CTYPE3'] = ('RGB', 'Color Format')
    else: # Grayscale
        image_to_save = image_uint16
        final_header['NAXIS'] = 2; final_header['NAXIS1'] = image.shape[1]
        final_header['NAXIS2'] = image.shape[0]
        if 'NAXIS3' in final_header: del final_header['NAXIS3']
        if 'CTYPE3' in final_header: del final_header['CTYPE3']

    final_header['BITPIX'] = 16; final_header['BSCALE'] = 1; final_header['BZERO'] = 32768

    # --- Write the FITS file WITH warning suppression ---
    
    try:
        hdu = fits.PrimaryHDU(data=image_to_save, header=final_header)
        hdul = fits.HDUList([hdu])

        # --- Start of the fix ---
        with warnings.catch_warnings():
            # Filter settings *inside* the 'with' block
            warnings.filterwarnings(
                'ignore',
                category=VerifyWarning,
                message="Keyword name.*is greater than 8 characters.*"
            )
            warnings.filterwarnings(
                'ignore',
                category=VerifyWarning,
                message="Keyword name.*contains characters not allowed.*"
            )

            # The actual saving function call is *inside* the 'with' block
            hdul.writeto(output_path, overwrite=overwrite, checksum=True)
        
    except Exception as e:
        print(f"Error saving FITS file to {output_path}: {e}")
        # Optional: Add traceback print here if needed for debugging save errors
        # import traceback
        # traceback.print_exc(limit=2)
        # raise # Re-raise if you want saving errors to stop the process

# ... (save_preview_image function) ...

        
def save_preview_image(image, output_path, apply_stretch=False, enhanced_stretch=False, color_balance=(1.0, 1.0, 1.0)):
    """
    Enregistre une image PNG/JPG pour la prévisualisation (normalisée 0-255 uint8).
    Applique potentiellement un étirement et une balance des blancs pour l'aperçu.

    Parameters:
        image (numpy.ndarray): Image 2D (HxW) ou 3D (HxWx3) à sauvegarder (float32, 0-1 range).
        output_path (str): Chemin du fichier PNG/JPG de sortie.
        apply_stretch (bool): Applique un étirement automatique si True.
        enhanced_stretch (bool): Si True et apply_stretch est True, utilise l'étirement amélioré.
        color_balance (tuple): Facteurs (R, G, B) pour la balance des blancs de l'aperçu.
    """
    if image is None:
        print(f"Warning: Attempted to save preview for None image to {output_path}")
        return

    preview = image.astype(np.float32).copy() # Work on a float32 copy

    # --- Apply Color Balance (only for color images) ---
    if preview.ndim == 3 and preview.shape[-1] == 3:
        try:
            # Import within function to potentially resolve circular dependency issues
            from ..tools.stretch import ColorCorrection
            preview = ColorCorrection.white_balance(preview, *color_balance)
        except ImportError:
             print("Warning: Could not import ColorCorrection for preview balance.")
        except Exception as wb_err:
             print(f"Warning: Failed to apply color balance to preview: {wb_err}")


    # --- Apply Stretch ---
    if apply_stretch:
        try:
            # Determine which stretch function to use
            if enhanced_stretch:
                from ..tools.stretch import apply_enhanced_stretch
                preview = apply_enhanced_stretch(preview) # Returns 0-1 float
            else:
                # Use a simple linear stretch based on percentiles for basic preview
                from ..tools.stretch import apply_auto_stretch, StretchPresets
                bp, wp = apply_auto_stretch(preview)
                preview = StretchPresets.linear(preview, bp, wp)

            preview = np.clip(preview, 0.0, 1.0) # Ensure stretch stays in 0-1

        except ImportError:
            print("Warning: Could not import stretch tools for preview generation.")
        except Exception as e:
            print(f"Error during preview stretch: {e}, saving non-stretched version.")
            # Fallback: use the original (potentially color-balanced) data without stretch

    # --- Convert to uint8 for saving ---
    # Scale float (0-1) to uint8 (0-255)
    preview_uint8 = (np.clip(preview, 0.0, 1.0) * 255.0).astype(np.uint8)

    # --- Handle color conversion for OpenCV saving (expects BGR) ---
    if preview_uint8.ndim == 3 and preview_uint8.shape[-1] == 3:
        # Input 'preview' was HxWxRGB (float), preview_uint8 is HxWxRGB (uint8)
        # Convert RGB to BGR for cv2.imwrite
        preview_bgr = cv2.cvtColor(preview_uint8, cv2.COLOR_RGB2BGR)
    elif preview_uint8.ndim == 2:
        # Grayscale image, OpenCV handles it directly
        preview_bgr = preview_uint8
    else:
        print(f"Error: Cannot save preview with shape {preview_uint8.shape}")
        return

    # --- Ensure output directory exists ---
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir: # Only create if path includes a directory
             os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory for preview {output_path}: {e}")
        return # Cannot save if directory fails

    # --- Save the image ---
    try:
        success = cv2.imwrite(output_path, preview_bgr)
        if not success:
             print(f"Error: cv2.imwrite failed to save preview image to {output_path}")
    except Exception as e:
        print(f"Error saving preview image to {output_path}: {e}")
# --- END OF FILE seestar/core/image_processing.py ---