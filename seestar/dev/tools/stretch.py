# --- START OF FILE seestar/tools/stretch.py ---
"""
Module contenant les algorithmes d'étirement d'histogramme et de correction couleur
inspirés de visu.py et adaptés pour Seestar Stacker.
"""

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageOps # Added ImageOps
import os # For save_fits_as_png
import traceback # For error reporting

# --- Presets d'Étirement ---
class StretchPresets:
    """ Fonctions d'étirement d'histogramme statiques.
        Attend des données d'entrée normalisées (0-1 float) et retourne (0-1 float).
    """
    @staticmethod
    def linear(data, bp=0., wp=1.):
        """Étirement linéaire simple."""
        if data is None: return None
        wp = max(float(wp), float(bp) + 1e-6) # Ensure wp > bp and float
        bp = float(bp)
        # Ensure data is float for calculation
        stretched = (data.astype(np.float32) - bp) / (wp - bp)
        return np.clip(stretched, 0.0, 1.0)

    @staticmethod
    def logarithmic(data, scale=10.0, bp=0.):
        """Étirement logarithmique."""
        if data is None: return None
        scale = float(scale); bp = float(bp)
        # Ensure data is float, handle non-finite values before processing
        data_float = np.nan_to_num(data.astype(np.float32))
        data_shifted = data_float - bp
        data_clipped = np.maximum(data_shifted, 1e-10) # Avoid log(0)
        max_val = np.nanmax(data_clipped)
        if max_val <= 0: return np.zeros_like(data_float)

        denominator = np.log1p(scale * max_val)
        if denominator < 1e-10: return np.zeros_like(data_float)

        stretched = np.log1p(scale * data_clipped) / denominator
        return np.clip(stretched, 0.0, 1.0)

    @staticmethod
    def asinh(data, scale=10.0, bp=0.):
        """Étirement Asinh (Arcsinus hyperbolique)."""
        if data is None: return None
        scale = float(scale); bp = float(bp)
        data_float = np.nan_to_num(data.astype(np.float32))
        data_shifted = data_float - bp
        data_clipped = np.maximum(data_shifted, 0.) # Asinh handles 0
        max_val = np.nanmax(data_clipped)
        # Check max_val after calculation
        if max_val <= 0: return np.zeros_like(data_float)

        denominator = np.arcsinh(scale * max_val)
        if denominator < 1e-10: return np.zeros_like(data_float)

        stretched = np.arcsinh(scale * data_clipped) / denominator
        return np.clip(stretched, 0.0, 1.0)

    @staticmethod
    def gamma(data, gamma=1.0):
        """Applique une correction Gamma."""
        if data is None: return None
        gamma = float(gamma)
        if abs(gamma - 1.0) < 1e-6: return data # Avoid calculation if gamma is 1
        # Avoid power(0, gamma < 1) issues and ensure float
        data_float = np.nan_to_num(data.astype(np.float32))
        corrected = np.power(np.maximum(data_float, 1e-10), gamma)
        return np.clip(corrected, 0.0, 1.0)


# --- Correction Couleur ---
class ColorCorrection:
    """ Fonctions de correction couleur statiques.
        Attend des données d'entrée normalisées (0-1 float) et retourne (0-1 float).
    """
    @staticmethod
    def white_balance(data, r=1., g=1., b=1.):
        """Balance des blancs simple par multiplication des canaux."""
        if data is None or data.ndim != 3 or data.shape[2] != 3:
            return data # Return original if not color or None
        # Ensure gains are floats
        r_gain, g_gain, b_gain = float(r), float(g), float(b)
        # Work on a float32 copy
        corrected = data.astype(np.float32, copy=True)
        corrected[..., 0] *= r_gain
        corrected[..., 1] *= g_gain
        corrected[..., 2] *= b_gain
        return np.clip(corrected, 0.0, 1.0)

# --- Fonctions d'aide Auto ---
def apply_auto_stretch(data):
    """
    Calcule des points noir et blanc automatiques basés sur les percentiles.

    Args:
        data (np.ndarray): Image (HxW ou HxWx3, float, 0-1) après balance des blancs.

    Returns:
        tuple: (black_point, white_point) calculés.
    """
    if data is None or data.size == 0: return (0.0, 1.0)

    try:
        if data.ndim == 3 and data.shape[2] == 3:
            # Utiliser luminance pour images couleur
            luminance = 0.299 * data[..., 0] + 0.587 * data[..., 1] + 0.114 * data[..., 2]
        elif data.ndim == 2:
            luminance = data
        else: return (0.0, 1.0) # Format non supporté

        # Filtrer NaN/Inf avant calculs
        finite_lum = luminance[np.isfinite(luminance)]
        if finite_lum.size < 20: print("Warning AutoStretch: Not enough finite pixels."); return (0.0, 1.0) # Besoin d'assez de points

        # Calculer les percentiles
        bp = np.percentile(finite_lum, 1.0)  # Point noir au 1er percentile
        wp = np.percentile(finite_lum, 99.0) # Point blanc au 99ème percentile

        # S'assurer que les points sont valides et distincts
        min_separation = 1e-4
        bp = np.clip(bp, 0.0, 1.0 - min_separation)
        wp = np.clip(wp, bp + min_separation, 1.0)

        # print(f"Auto Stretch calculated: BP={bp:.4f}, WP={wp:.4f}") # Debug
        return bp, wp

    except Exception as e:
        print(f"Error during Auto Stretch calculation: {e}")
        traceback.print_exc(limit=1)
        return (0.0, 1.0) # Fallback


def apply_auto_white_balance(data):
    """
    Calcule les gains R, G, B pour une balance des blancs automatique simple
    basée sur l'égalisation des modes des canaux (style visu.py).

    Args:
        data (np.ndarray): Image couleur (HxWx3, float, 0-1).

    Returns:
        tuple: (r_gain, g_gain, b_gain) calculés.
    """
    if data is None or data.ndim != 3 or data.shape[2] != 3:
        return (1.0, 1.0, 1.0) # Return default gains if not color or None

    try:
        modes = []
        num_bins = 256
        for i in range(3): # R, G, B channels
            channel_data = data[..., i].ravel()
            finite_data = channel_data[np.isfinite(channel_data)]
            if finite_data.size == 0: raise ValueError(f"Channel {i} is empty or all NaN/Inf.")

            # Calculer le mode basé sur l'histogramme des valeurs centrales
            min_r, max_r = np.percentile(finite_data, [0.5, 99.5]) # Ignorer extrêmes
            if max_r <= min_r: max_r = min_r + 1e-5 # Assurer une plage valide

            hist, bin_edges = np.histogram(finite_data, bins=num_bins, range=(min_r, max_r))
            mode_index = np.argmax(hist)
            # Prendre le centre du bin modal
            channel_mode = (bin_edges[mode_index] + bin_edges[mode_index+1]) / 2
            channel_mode = max(channel_mode, 1e-5) # Eviter mode nul
            modes.append(channel_mode)

        mode_r, mode_g, mode_b = modes
        # print(f"Auto WB Modes: R={mode_r:.4f}, G={mode_g:.4f}, B={mode_b:.4f}") # Debug

        # Calculer gains pour égaliser les modes (cible = mode vert)
        gain_r = mode_g / mode_r if mode_r > 1e-9 else 1.0
        gain_g = 1.0
        gain_b = mode_g / mode_b if mode_b > 1e-9 else 1.0

        # Limiter les gains pour éviter des couleurs extrêmes
        max_gain = 5.0; min_gain = 0.2
        gain_r = np.clip(gain_r, min_gain, max_gain)
        gain_b = np.clip(gain_b, min_gain, max_gain)

        # print(f"Auto WB Gains: R={gain_r:.3f}, G={gain_g:.3f}, B={gain_b:.3f}") # Debug
        return gain_r, gain_g, gain_b

    except Exception as e:
        print(f"Error during Auto WB calculation: {e}")
        traceback.print_exc(limit=2)
        return (1.0, 1.0, 1.0) # Fallback


# --- Enhanced Stretch (Example implementation) ---
def apply_enhanced_stretch(data, saturation=1.2, clahe_strength=2.0, clahe_tile_size=8, sharpen=False):
    """
    Applique un étirement amélioré combinant Asinh stretch,
    saturation, et amélioration locale du contraste (CLAHE).

    Args:
        data (np.ndarray): Image (HxW ou HxWx3, float, 0-1).
        saturation (float): Facteur d'amélioration de la saturation (1.0 = pas de changement).
        clahe_strength (float): Limite de clip pour CLAHE (typiquement 1.0-4.0).
        clahe_tile_size (int): Taille de la grille pour CLAHE (typiquement 8).
        sharpen (bool): Appliquer un léger filtre de netteté.

    Returns:
        np.ndarray: Image améliorée (float, 0-1). Returns input data if error occurs.
    """
    if data is None: return None

    try:
        stretched_data = data.copy()

        # 1. Apply a base stretch (e.g., Asinh with auto levels)
        bp, wp = apply_auto_stretch(stretched_data)
        asinh_scale = 10.0 / max(0.01, wp - bp) if wp > bp else 10.0
        stretched_data = StretchPresets.asinh(stretched_data, scale=asinh_scale, bp=bp)
        stretched_data = np.clip(stretched_data, 0.0, 1.0)

        pil_img_uint8 = (np.nan_to_num(stretched_data) * 255).astype(np.uint8)

        is_color = False
        if pil_img_uint8.ndim == 2:
            pil_img = Image.fromarray(pil_img_uint8, mode='L')
        elif pil_img_uint8.ndim == 3 and pil_img_uint8.shape[2] == 3:
            pil_img = Image.fromarray(pil_img_uint8, mode='RGB')
            is_color = True
        else:
            print("Warning: Cannot apply enhanced stretch to this image format.")
            return data # Return original data

        # 2. Apply CLAHE
        if clahe_strength > 0 and clahe_tile_size > 1:
            try:
                img_for_clahe = np.array(pil_img)
                if is_color:
                    # Convert to LAB color space for CLAHE on L channel
                    lab = cv2.cvtColor(img_for_clahe, cv2.COLOR_RGB2LAB)
                    l_channel, a_channel, b_channel = cv2.split(lab)
                    clahe_processor = cv2.createCLAHE(clipLimit=float(clahe_strength), tileGridSize=(int(clahe_tile_size), int(clahe_tile_size)))
                    cl = clahe_processor.apply(l_channel)
                    limg = cv2.merge((cl, a_channel, b_channel))
                    enhanced_img_uint8 = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
                else: # Grayscale
                    clahe_processor = cv2.createCLAHE(clipLimit=float(clahe_strength), tileGridSize=(int(clahe_tile_size), int(clahe_tile_size)))
                    enhanced_img_uint8 = clahe_processor.apply(img_for_clahe)
                pil_img = Image.fromarray(enhanced_img_uint8)
            except Exception as e:
                print(f"Warning: CLAHE application failed: {e}")

        # 3. Enhance Saturation
        if is_color and abs(saturation - 1.0) > 1e-2:
            try:
                enhancer = ImageEnhance.Color(pil_img)
                pil_img = enhancer.enhance(float(saturation))
            except Exception as e:
                print(f"Warning: Saturation enhancement failed: {e}")

        # 4. Apply Sharpening
        if sharpen:
            try:
                pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
            except Exception as e:
                 print(f"Warning: Sharpening failed: {e}")

        # 5. Convert final PIL image back to numpy float32 (0-1)
        final_data_float = np.array(pil_img).astype(np.float32) / 255.0

        return final_data_float

    except Exception as e:
         print(f"Error in apply_enhanced_stretch: {e}")
         traceback.print_exc(limit=2)
         return data # Return original data on error


# --- Helper for Saving Stretched Image as PNG/JPG ---
def save_fits_as_png(data, output_path, enhance=False, color_balance=(1.0,1.0,1.0)):
    """
    Applique un stretch (optionnellement amélioré) et sauvegarde en PNG/JPG.

    Args:
        data (numpy.ndarray): Données FITS (HxW ou HxWx3, float, 0-1).
        output_path (str): Chemin de sortie (.png ou .jpg).
        enhance (bool): Utiliser l'étirement amélioré.
        color_balance (tuple): Facteurs (R,G,B) pour la balance des blancs.
    """
    if data is None:
        print(f"Error: Cannot save None data to {output_path}")
        return False

    try:
        processed_data = data.copy()

        # Apply color balance first
        if processed_data.ndim == 3 and processed_data.shape[2] == 3:
             processed_data = ColorCorrection.white_balance(processed_data, *color_balance)

        # Apply stretch
        if enhance:
            stretched_data = apply_enhanced_stretch(processed_data)
        else:
            bp, wp = apply_auto_stretch(processed_data)
            stretched_data = StretchPresets.linear(processed_data, bp, wp)

        # Convert final stretched data (0-1 float) to uint8 for saving
        img_uint8 = (np.clip(stretched_data, 0.0, 1.0) * 255).astype(np.uint8)

        # Create PIL image from uint8 data
        if img_uint8.ndim == 3 and img_uint8.shape[2] == 3:
             pil_image = Image.fromarray(img_uint8, 'RGB')
        elif img_uint8.ndim == 2:
             pil_image = Image.fromarray(img_uint8, 'L').convert('RGB') # Save grayscale as RGB PNG
        else:
             print(f"Error: Cannot save image with shape {img_uint8.shape} as PNG/JPG.")
             return False

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
             os.makedirs(output_dir, exist_ok=True)

        # Save using PIL
        pil_image.save(output_path)
        return True

    except Exception as e:
        print(f"Error saving preview image to {output_path}: {e}")
        traceback.print_exc(limit=2)
        return False
# --- END OF FILE seestar/tools/stretch.py ---