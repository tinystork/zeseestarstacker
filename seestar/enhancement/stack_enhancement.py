# --- START OF FILE seestar/enhancement/stack_enhancement.py (MODIFIED) ---
import cv2
import numpy as np
from skimage import exposure
import time
import traceback # Garder pour les logs d'erreur

# Import ccdproc SEULEMENT si on décide de garder une fonction de combine ici
# Pour l'instant, on suppose que ccdproc est utilisé dans queue_manager
# try:
#     from ccdproc import combine, CCDData
# except ImportError:
#     print("ERREUR CRITIQUE: Impossible d'importer 'combine' ou 'CCDData' depuis ccdproc.")
#     print("Veuillez installer ou mettre à jour ccdproc: pip install ccdproc")
#     raise ImportError("ccdproc not found or combine/CCDData missing")

# Import du DrizzleProcessor modifié (qui utilise stsci.drizzle)
try:
    from .drizzle_integration import DrizzleProcessor # Utiliser chemin relatif
    _drizzle_available = True
except ImportError:
    print("StackEnhancer: Module drizzle_integration non trouvé. L'option Drizzle sera désactivée.")
    _drizzle_available = False
    DrizzleProcessor = None # Définir comme None si non disponible

import traceback # Pour un meilleur débogage

class StackEnhancer:
    def __init__(self, config=None):
        """
        Initialise l'enhancer de stack.

        Args:
            config (dict, optional): Dictionnaire de configuration.
                                    Peut contenir 'drizzle_scale', 'drizzle_pixfrac',
                                    'normalization', 'clahe_params', 'edge_crop_percent'.
        """
#############################################################################REGLAGE ROGNAGEEFFET DE BORD########################################### 
        default_config = {
            'drizzle_scale': 2.0,       # Facteur d'échelle Drizzle
            'drizzle_pixfrac': 1.0,     # Fraction de pixel Drizzle
            'normalization': 'skimage', # 'astropy' | 'skimage' | 'basic' | 'none' <-- Ajouté 'none'
            'clahe_params': {'clip_limit': 2.0, 'tile_grid_size': (8, 8)},
            'edge_crop_percent': 0.3   # % à rogner sur chaque bord (0 = pas de rognage)
        }

####################################################################################################################################################
        self.config = default_config
        if config:
            try:
                if 'drizzle_scale' in config: self.config['drizzle_scale'] = float(config['drizzle_scale'])
                if 'drizzle_pixfrac' in config: self.config['drizzle_pixfrac'] = float(config['drizzle_pixfrac'])
                # Accepter 'none' pour la normalisation
                if 'normalization' in config and config['normalization'] in ['astropy', 'skimage', 'basic', 'none']:
                    self.config['normalization'] = str(config['normalization'])
                elif 'normalization' in config:
                    print(f"Warning: Invalid normalization method '{config['normalization']}'. Using default.")
                if 'clahe_params' in config and isinstance(config['clahe_params'], dict):
                    self.config['clahe_params']['clip_limit'] = float(config['clahe_params'].get('clip_limit', default_config['clahe_params']['clip_limit']))
                    grid_size = config['clahe_params'].get('tile_grid_size', default_config['clahe_params']['tile_grid_size'])
                    if isinstance(grid_size, (list, tuple)) and len(grid_size) == 2:
                        self.config['clahe_params']['tile_grid_size'] = (int(grid_size[0]), int(grid_size[1]))
                    else: print(f"Warning: Invalid tile_grid_size ({grid_size}). Using default.")
                if 'edge_crop_percent' in config: self.config['edge_crop_percent'] = float(config['edge_crop_percent'])
            except (ValueError, TypeError) as e:
                print(f"Warning: Invalid value in StackEnhancer config: {e}. Using defaults where necessary.")

        # Validation finale des valeurs de config (inchangée)
        if not (0.0 <= self.config['edge_crop_percent'] < 0.5):
            print(f"Warning: Invalid edge_crop_percent ({self.config['edge_crop_percent']}). Setting to 0.02.")
            self.config['edge_crop_percent'] = 0.02
        if not (0.0 < self.config['drizzle_pixfrac'] <= 1.0):
             print(f"Warning: Invalid drizzle_pixfrac ({self.config['drizzle_pixfrac']}). Setting to 1.0.")
             self.config['drizzle_pixfrac'] = 1.0
        if self.config['drizzle_scale'] < 1.0:
             print(f"Warning: Invalid drizzle_scale ({self.config['drizzle_scale']}). Setting to 2.0.")
             self.config['drizzle_scale'] = 2.0


    def _normalize_stack(self, images):
        """
        Normalisation scientifique des images (suppose float32 0-1 en entrée).
        Retourne des images potentiellement hors [0,1] pour 'astropy'.
        Si method='none', retourne les images originales.
        """
        method = self.config['normalization']
        # Ne pas logger si 'none'
        if method != 'none':
            print(f"... Normalisation des images (méthode: {method})...")
        normalized_images = []
        for i, img in enumerate(images):
            try:
                if img is None: continue # Ignorer images None
                if img.dtype != np.float32: img = img.astype(np.float32)

                # --- Ajout de la condition 'none' ---
                if method == 'none':
                    normalized_images.append(img) # Retourne l'original
                    continue
                # --- Fin ajout ---

                if method == 'astropy':
                    # Normalisation par médiane et MAD robuste (comme sigma clipping)
                    median = np.nanmedian(img)
                    mad = np.nanmedian(np.abs(img - median))
                    sigma_robust = max(mad * 1.4826, 1e-9)
                    normalized = (img - median) / sigma_robust
                    normalized_images.append(normalized)
                elif method == 'skimage':
                    normalized = exposure.rescale_intensity(img, out_range=(0.0, 1.0))
                    normalized_images.append(normalized.astype(np.float32))
                else: # Méthode 'basic' ou inconnue -> Min-Max standard
                    min_val, max_val = np.nanmin(img), np.nanmax(img)
                    if max_val > min_val:
                        normalized = (img - min_val) / (max_val - min_val)
                    else: # Image constante
                        normalized = np.zeros_like(img, dtype=np.float32)
                    normalized_images.append(np.clip(normalized, 0.0, 1.0)) # Assurer [0,1]

            except Exception as e:
                print(f"   -> ERREUR normalisation image {i}: {e}")
                traceback.print_exc(limit=1)

        # Ne pas logger si 'none'
        if method != 'none':
             print(f"... Normalisation terminée ({len(normalized_images)} images).")
        return normalized_images

    def _apply_clahe(self, img):
        """
        Égalisation adaptative CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Applique sur la luminance (L*) pour les images couleur (RGB -> LAB -> RGB).
        ATTENTION: Attend une image en float32 [0, 1].
                   Retourne une image en float32 [0, 1].
        """
        if img is None: return None
        print("... Application CLAHE...") # Garder ce log

        # Convertir l'image float [0,1] en uint8 [0,255] pour OpenCV CLAHE
        img_uint8 = (np.clip(np.nan_to_num(img), 0.0, 1.0) * 255.0).astype(np.uint8)

        try:
            clahe = cv2.createCLAHE(
                clipLimit=self.config['clahe_params']['clip_limit'],
                tileGridSize=self.config['clahe_params']['tile_grid_size']
            )

            if img_uint8.ndim == 3 and img_uint8.shape[2] == 3: # Couleur
                lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                cl = clahe.apply(l)
                limg = cv2.merge((cl, a, b))
                final_uint8 = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            elif img_uint8.ndim == 2: # Grayscale
                final_uint8 = clahe.apply(img_uint8)
            else:
                print("   -> Warning CLAHE: Format d'image non supporté.")
                return img # Retourner l'original si format inconnu

            print("... CLAHE terminé.")
            return final_uint8.astype(np.float32) / 255.0

        except cv2.error as cv_err:
             print(f"   -> ERREUR OpenCV pendant CLAHE: {cv_err}")
             return img # Retourner l'original en cas d'erreur CLAHE
        except Exception as e:
             print(f"   -> ERREUR inattendue pendant CLAHE: {e}")
             traceback.print_exc(limit=1)
             return img


def apply_edge_crop(img_data, crop_percent_decimal):
    """
    Applique un rognage en pourcentage sur les bords d'une image.

    Args:
        img_data (np.ndarray): Image à rogner.
        crop_percent_decimal (float): Pourcentage à rogner (ex: 0.05 pour 5%).

    Returns:
        np.ndarray: Image rognée, ou l'originale si rognage non applicable/erreur.
    """
    if img_data is None:
        print("WARN [apply_edge_crop]: Données image None, pas de rognage.")
        return None
    if not (isinstance(crop_percent_decimal, (float, int)) and 0 < crop_percent_decimal < 0.5):
        # print("DEBUG [apply_edge_crop]: Pourcentage de rognage invalide ou nul, pas de rognage.")
        return img_data # Pas de rognage si 0 ou invalide

    print(f"DEBUG [apply_edge_crop]: Rognage des bords demandé ({crop_percent_decimal*100:.1f}%)...")
    try:
        h, w = img_data.shape[:2]
        crop_h_pixels = int(h * crop_percent_decimal)
        crop_w_pixels = int(w * crop_percent_decimal)

        if (crop_h_pixels * 2 >= h) or (crop_w_pixels * 2 >= w):
            print(f"   WARN [apply_edge_crop]: Rognage excessif demandé. Annulé.")
            return img_data

        if crop_h_pixels > 0 or crop_w_pixels > 0:
            y_start, y_end = crop_h_pixels, h - crop_h_pixels
            x_start, x_end = crop_w_pixels, w - crop_w_pixels
            
            cropped_img = None
            if img_data.ndim == 3:
                cropped_img = img_data[y_start:y_end, x_start:x_end, :].copy() # .copy() pour éviter problèmes de vue
            elif img_data.ndim == 2:
                cropped_img = img_data[y_start:y_end, x_start:x_end].copy()
            else:
                print(f"   WARN [apply_edge_crop]: Format image inattendu ({img_data.ndim}D). Pas de rognage.")
                return img_data
            
            print(f"DEBUG [apply_edge_crop]: Rognage terminé. Nouvelle shape: {cropped_img.shape}.")
            return cropped_img
        else:
            # print("DEBUG [apply_edge_crop]: Pixels de rognage calculés à 0, pas de rognage effectué.")
            return img_data # Pas de rognage si pixels = 0

    except Exception as e:
         print(f"   ERREUR [apply_edge_crop]: Erreur pendant le rognage: {e}")
         traceback.print_exc(limit=1)
         return img_data # Retourner l'original en cas d'erreur

# DANS stack_enhancement.py (ou où vous l'avez mise)

def feather_by_weight_map(img, wht, blur_px=256, eps=1e-6, min_gain=0.5, max_gain=2.0): # Ajout min/max_gain
    print(f"DEBUG [feather_by_weight_map]: Début. ImgS: {img.shape}, WHTS: {wht.shape}, blur: {blur_px}, minG: {min_gain}, maxG: {max_gain}")
    # ... (vérifications initiales img, wht, etc. inchangées) ...
    if img is None or wht is None: return img
    if img.ndim != 3 or img.shape[2] != 3 or wht.ndim != 2 or img.shape[:2] != wht.shape: return img

    try:
        img_f32 = img.astype(np.float32, copy=False)
        wht_f32 = wht.astype(np.float32, copy=False)
        blur_px = max(1, int(blur_px)); kernel_size = (blur_px // 2) * 2 + 1
        print(f"DEBUG [feather_by_weight_map]: Kernel: {kernel_size}x{kernel_size}")

        wht_blurred = cv2.GaussianBlur(wht_f32, (kernel_size, kernel_size), 0)
        wht_min_for_gain = np.percentile(wht_f32[wht_f32 > eps], 1) # Prendre le 1er percentile des poids non nuls comme seuil min
        wht_min_for_gain = max(wht_min_for_gain, eps * 10) # Assurer qu'il est un peu au-dessus de epsilon
        
        gain_map = wht_blurred / np.maximum(wht_f32, wht_min_for_gain) # Utiliser un wht minimum plus élevé
        
        # --- CLIPPING DU GAIN ---
        gain_map_clipped = np.clip(gain_map, min_gain, max_gain)
        print(f"DEBUG [feather_by_weight_map]: GainMapNonClipped Range: [{np.min(gain_map):.2f}-{np.max(gain_map):.2f}]")
        print(f"DEBUG [feather_by_weight_map]: GainMapClipped   Range: [{np.min(gain_map_clipped):.2f}-{np.max(gain_map_clipped):.2f}]")
        
        gain_map_blurred = cv2.GaussianBlur(gain_map_clipped, (kernel_size, kernel_size), 0) # Flouter le gain clippé
        print(f"DEBUG [feather_by_weight_map]: GainMapBlurred(Clipped) Range: [{np.min(gain_map_blurred):.2f}-{np.max(gain_map_blurred):.2f}]")

        feathered_image = img_f32 * gain_map_blurred[..., None]
        feathered_image_clipped = np.clip(feathered_image, 0., 1.)
        return feathered_image_clipped.astype(np.float32)
    except Exception as e:
        print(f"ERREUR [feather_by_weight_map]: {e}"); traceback.print_exc(limit=2)
        return img




# DANS stack_enhancement.py (ou où vous l'avez mise)

def feather_by_weight_map(img, wht, blur_px=256, eps=1e-6, min_gain=0.5, max_gain=2.0): # Ajout min/max_gain
    print(f"DEBUG [feather_by_weight_map]: Début. ImgS: {img.shape}, WHTS: {wht.shape}, blur: {blur_px}, minG: {min_gain}, maxG: {max_gain}")
    # ... (vérifications initiales img, wht, etc. inchangées) ...
    if img is None or wht is None: return img
    if img.ndim != 3 or img.shape[2] != 3 or wht.ndim != 2 or img.shape[:2] != wht.shape: return img

    try:
        img_f32 = img.astype(np.float32, copy=False)
        wht_f32 = wht.astype(np.float32, copy=False)
        blur_px = max(1, int(blur_px)); kernel_size = (blur_px // 2) * 2 + 1
        print(f"DEBUG [feather_by_weight_map]: Kernel: {kernel_size}x{kernel_size}")

        wht_blurred = cv2.GaussianBlur(wht_f32, (kernel_size, kernel_size), 0)
        wht_min_for_gain = np.percentile(wht_f32[wht_f32 > eps], 1) # Prendre le 1er percentile des poids non nuls comme seuil min
        wht_min_for_gain = max(wht_min_for_gain, eps * 10) # Assurer qu'il est un peu au-dessus de epsilon
        
        gain_map = wht_blurred / np.maximum(wht_f32, wht_min_for_gain) # Utiliser un wht minimum plus élevé
        
        # --- CLIPPING DU GAIN ---
        gain_map_clipped = np.clip(gain_map, min_gain, max_gain)
        print(f"DEBUG [feather_by_weight_map]: GainMapNonClipped Range: [{np.min(gain_map):.2f}-{np.max(gain_map):.2f}]")
        print(f"DEBUG [feather_by_weight_map]: GainMapClipped   Range: [{np.min(gain_map_clipped):.2f}-{np.max(gain_map_clipped):.2f}]")
        
        gain_map_blurred = cv2.GaussianBlur(gain_map_clipped, (kernel_size, kernel_size), 0) # Flouter le gain clippé
        print(f"DEBUG [feather_by_weight_map]: GainMapBlurred(Clipped) Range: [{np.min(gain_map_blurred):.2f}-{np.max(gain_map_blurred):.2f}]")

        feathered_image = img_f32 * gain_map_blurred[..., None]
        feathered_image_clipped = np.clip(feathered_image, 0., 1.)
        return feathered_image_clipped.astype(np.float32)
    except Exception as e:
        print(f"ERREUR [feather_by_weight_map]: {e}"); traceback.print_exc(limit=2)
        return img




# --- NOUVELLE MÉTHODE pour appliquer post-traitement seul ---
def apply_postprocessing(self, image_data):
    """
    Applique uniquement les étapes de post-traitement (rognage, CLAHE)
    à une image déjà combinée.

    Args:
        image_data (np.ndarray): Image combinée (float32, 0-1).

    Returns:
        np.ndarray: Image post-traitée (float32, 0-1).
    """
    if image_data is None:
        print("StackEnhancer.apply_postprocessing: Aucune donnée fournie.")
        return None

    print("StackEnhancer: Application post-traitement (Rognage, CLAHE)...")
    processed_data = image_data.copy() # Travailler sur une copie

    # 1. Rognage basé sur le pourcentage de la config
    processed_data = self._postprocess_edges(processed_data)
    if processed_data is None: return image_data # Retourner l'original si rognage échoue

    # 2. CLAHE (Optionnel - Décommenter pour activer)
    # print("... Application CLAHE DÉSACTIVÉE pour test (dans apply_postprocessing)...")
    # processed_data = self._apply_clahe(processed_data)
    # if processed_data is None: return image_data # Retourner l'original si CLAHE échoue

    print("StackEnhancer: Post-traitement terminé.")
    return processed_data.astype(np.float32) # Assurer float32
# --- FIN NOUVELLE MÉTHODE ---

def _postprocess_edges(self, img):
    """
    Réduction des artefacts de bord par rognage simple basé sur un pourcentage.
    (Méthode inchangée)
    """
    if img is None:
        print("   -> Warning: _postprocess_edges reçu None.")
        return None

    crop_percent = self.config.get('edge_crop_percent', 0.02)

    if not isinstance(crop_percent, (float, int)) or crop_percent <= 0.0:
        return img # Pas de rognage

    print(f"... Rognage des bords ({crop_percent*100:.1f}%)...")
    try:
        h, w = img.shape[:2]
        crop_h_pixels = int(h * crop_percent)
        crop_w_pixels = int(w * crop_percent)

        if (crop_h_pixels * 2 >= h) or (crop_w_pixels * 2 >= w):
            print(f"   -> Warning: Rognage excessif demandé ({crop_percent*100:.1f}%). Rognage annulé.")
            return img

        if crop_h_pixels > 0 or crop_w_pixels > 0:
            y_start, y_end = crop_h_pixels, h - crop_h_pixels
            x_start, x_end = crop_w_pixels, w - crop_w_pixels
            if img.ndim == 3: cropped_img = img[y_start:y_end, x_start:x_end, :]
            elif img.ndim == 2: cropped_img = img[y_start:y_end, x_start:x_end]
            else: print(f"   -> Warning: Format d'image inattendu ({img.ndim}D) pour le rognage."); return img
            print(f"... Rognage terminé (Nouvelle shape: {cropped_img.shape}).")
            return cropped_img
        else:
            return img

    except Exception as e:
            print(f"   -> ERREUR pendant le rognage: {e}")
            traceback.print_exc(limit=1)
            return img # Retourner l'original en cas d'erreur
# --- END OF FILE seestar/enhancement/stack_enhancement.py (MODIFIED) ---