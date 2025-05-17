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
import sys
warnings.filterwarnings("ignore", category=FutureWarning)




# --- DANS seestar/core/image_processing.py ---

def load_and_validate_fits(filepath: str):
    """
    Charge un fichier FITS, le valide, le convertit en float32 et le normalise [0,1].
    Gère la conversion de type et les erreurs potentielles de manière robuste.
    Retourne également le header FITS original.
    MODIFIED: Logs de debug détaillés ajoutés, gestion des types améliorée.

    Args:
        filepath (str): Chemin complet vers le fichier FITS.

    Returns:
        tuple: (np.ndarray or None, fits.Header or None)
               - L'array image (H,W ou H,W,C) normalisé [0,1] float32.
               - Le header FITS original.
               Retourne (None, None) en cas d'erreur majeure non récupérable,
               ou (image_noire, header) si l'image est inutilisable mais que le header est lu.
    """
    base_filename = os.path.basename(filepath) # Pour les logs
    print(f"DEBUG IP (load_and_validate_fits V2): Début chargement pour '{base_filename}'")

    # --- 1. Vérification initiale du chemin ---
    if not filepath or not isinstance(filepath, str):
        print(f"  REJET (load_and_validate_fits V2): Chemin de fichier invalide fourni ({filepath}).")
        return None, None
    if not os.path.exists(filepath):
        print(f"  REJET (load_and_validate_fits V2): Fichier non trouvé à '{filepath}'.")
        return None, None

    # --- 2. Tentative de chargement FITS ---
    data_raw = None # Contiendra les données brutes lues du FITS
    header = None
    try:
        print(f"  DEBUG IP (load_and_validate_fits V2): Tentative fits.open pour '{base_filename}'...")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=AstropyWarning) # Ignore tous les warnings Astropy ici
            # do_not_scale_image_data=True est important pour lire les valeurs brutes
            with fits.open(filepath, memmap=False, do_not_scale_image_data=True) as hdul:
                image_hdu = None
                if len(hdul) == 0:
                    print(f"  REJET (load_and_validate_fits V2): Fichier FITS vide (0 HDUs) pour '{base_filename}'.")
                    return None, None

                # Essayer de trouver la première HDU avec des données image valides
                for i_hdu, hdu_item in enumerate(hdul):
                    if hdu_item.is_image and hdu_item.data is not None:
                        # Vérifier si les données ont au moins 2 dimensions (HxW)
                        if hdu_item.data.ndim >= 2:
                            image_hdu = hdu_item
                            print(f"  DEBUG IP (load_and_validate_fits V2): Données image trouvées dans HDU {i_hdu} pour '{base_filename}'. Shape brute: {hdu_item.data.shape}, Dtype brut: {hdu_item.data.dtype}")
                            break
                        else:
                            print(f"  WARN IP (load_and_validate_fits V2): HDU {i_hdu} est image mais data.ndim < 2 ({hdu_item.data.ndim}) pour '{base_filename}'. On continue la recherche.")
                
                if image_hdu is None:
                    print(f"  REJET (load_and_validate_fits V2): Aucune HDU image avec données >= 2D trouvée dans '{base_filename}'.")
                    # Essayer de retourner le header de la première HDU si possible pour logs
                    try: header_fallback = hdul[0].header.copy() if len(hdul) > 0 and hdul[0].header else None
                    except: header_fallback = None
                    return None, header_fallback
                
                data_raw = image_hdu.data 
                header = image_hdu.header.copy()

        if data_raw is None: 
            print(f"  REJET (load_and_validate_fits V2): Échec extraction données (data_raw est None) de '{base_filename}'.")
            return None, header # Retourner header même si data est None, pour logs

    except FileNotFoundError: # Devrait être attrapé par os.path.exists, mais sécurité
        print(f"  REJET (load_and_validate_fits V2): FileNotFoundError (devrait être impossible ici) pour '{filepath}'.")
        return None, None
    except Exception as e_load:
        print(f"  REJET (load_and_validate_fits V2): Exception lors du chargement FITS de '{base_filename}': {type(e_load).__name__} - {e_load}")
        return None, header 

    # --- 3. Validation des dimensions et type des données BRUTES ---
    print(f"  DEBUG IP (load_and_validate_fits V2): Validation données brutes pour '{base_filename}'. Shape: {data_raw.shape}, Dtype: {data_raw.dtype}")
    
    # La logique originale de rejet des shapes CxHxW reste, car cette fonction est pour les images "pixel-like"
    if data_raw.ndim not in [2, 3]: 
        print(f"  REJET (load_and_validate_fits V2): Shape de données brutes non supportée ({data_raw.shape}) pour '{base_filename}'. Doit être 2D (HxW) ou 3D (HxWxC).")
        return None, header
    if data_raw.ndim == 3 and data_raw.shape[-1] not in [1, 3, 4]: # HxWxC: C doit être 1 (gris), 3 (RGB) ou 4 (RGBA)
        # Si c'est CxHxW, data_raw.shape[-1] sera W, ce qui sera rejeté ici.
        print(f"  REJET (load_and_validate_fits V2): Shape de données brutes 3D non supportée ({data_raw.shape}) pour '{base_filename}'. Dernier axe (canaux) doit être 1, 3 ou 4. Reçu: {data_raw.shape[-1]}.")
        return None, header
    if data_raw.size == 0: 
        print(f"  REJET (load_and_validate_fits V2): Image brute vide (size=0) pour '{base_filename}'.")
        return None, header
        
    # --- 4. Conversion en float32 (si nécessaire) et Normalisation [0,1] ---
    print(f"  DEBUG IP (load_and_validate_fits V2): Préparation pour normalisation de '{base_filename}'...")
    
    # S'assurer qu'on travaille avec des flottants pour les stats min/max pour éviter overflow/underflow des entiers
    # et pour gérer les NaN potentiels.
    # Utiliser float64 pour les stats pour plus de précision, surtout avec des entiers larges.
    if data_raw.dtype not in [np.float32, np.float64]:
        print(f"    DEBUG IP: Conversion data_raw (dtype {data_raw.dtype}) vers float64 pour stats...")
        data_for_stats = data_raw.astype(np.float64)
    else:
        # Si déjà float, faire une copie pour ne pas modifier data_raw en place si c'est une vue
        data_for_stats = data_raw.copy().astype(np.float64) # Assurer float64 pour stats
    
    min_val_stats = np.nanmin(data_for_stats)
    max_val_stats = np.nanmax(data_for_stats)
    print(f"    DEBUG IP: Stats sur float64 pour '{base_filename}': Min={min_val_stats}, Max={max_val_stats} (dtype_stats: {data_for_stats.dtype})")

    if not (np.isfinite(min_val_stats) and np.isfinite(max_val_stats)):
        print(f"  WARN (load_and_validate_fits V2): Données FITS '{base_filename}' contiennent seulement NaN/Inf ou erreur de calcul min/max. Retourne image noire.")
        # Créer une image noire avec la même shape que data_raw
        return np.zeros_like(data_raw, dtype=np.float32), header

    denominator = max_val_stats - min_val_stats
    
    # Convertir les données originales en float32 pour la normalisation et la sortie finale.
    # Ceci crée une copie si data_raw n'est pas déjà float32.
    data_float32_to_norm = data_raw.astype(np.float32, copy=True) 
    print(f"    DEBUG IP: data_float32_to_norm (pour normalisation) - dtype: {data_float32_to_norm.dtype}, shape: {data_float32_to_norm.shape}")

    data_normalized = None # Initialiser
    if denominator > 1e-9: # Éviter division par zéro ou par un nombre très petit
        print(f"    DEBUG IP: Normalisation standard pour '{base_filename}' (denominator={denominator:.4g})...")
        data_normalized = (data_float32_to_norm - float(min_val_stats)) / float(denominator)
        data_normalized = np.clip(data_normalized, 0.0, 1.0) # Assurer [0,1]
    elif np.any(np.isfinite(data_float32_to_norm)): 
        data_normalized = np.full_like(data_float32_to_norm, 0.5, dtype=np.float32) 
        print(f"  WARN (load_and_validate_fits V2): Image '{base_filename}' quasi-constante (min~max). Normalisée à 0.5. MinVal={min_val_stats}, MaxVal={max_val_stats}")
    else: 
        data_normalized = np.zeros_like(data_float32_to_norm, dtype=np.float32) 
        print(f"  WARN (load_and_validate_fits V2): Image '{base_filename}' vide ou tout NaN/Inf (après cast float32). Normalisée à 0.0.")
    
    # Nettoyage final des NaN/Inf et s'assurer du type float32
    # Si data_normalized est None (ne devrait pas arriver avec la logique ci-dessus), nan_to_num plantera.
    if data_normalized is None: # Sécurité, ne devrait jamais être atteint.
        print(f"  CRITICAL ERROR IP: data_normalized est None AVANT nan_to_num pour '{base_filename}'. C'est un bug. Retourne None.")
        return None, header

    final_data = np.nan_to_num(data_normalized, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    
    # --- Log de la plage finale ---
    min_final, max_final = np.min(final_data), np.max(final_data) # Utiliser np.min/max car plus de NaN
    mean_final = np.mean(final_data)
    std_final = np.std(final_data) # Calculer l'écart-type final pour info
    print(f"  DEBUG IP (load_and_validate_fits V2): FIN pour '{base_filename}'. "
          f"Shape sortie: {final_data.shape}, Dtype: {final_data.dtype}. "
          f"Range final: [{min_final:.4f} - {max_final:.4f}], Moyenne: {mean_final:.4f}, StdDev: {std_final:.6f}")

    return final_data, header



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