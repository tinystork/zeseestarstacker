import cv2
import numpy as np
from skimage import exposure
import traceback # Garder pour les logs d'erreur


# ... (autres fonctions comme feather_by_weight_map, StackEnhancer, etc.) ...

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
            'normalization': 'skimage', # 'astropy' | 'skimage' | 'basic' | 'linear_fit' | 'sky_mean' | 'none'
            'clahe_params': {'clip_limit': 2.0, 'tile_grid_size': (8, 8)},
            'edge_crop_percent': 0.05   # % à rogner sur chaque bord (0 = pas de rognage)
        }

####################################################################################################################################################
        self.config = default_config
        if config:
            try:
                if 'drizzle_scale' in config: self.config['drizzle_scale'] = float(config['drizzle_scale'])
                if 'drizzle_pixfrac' in config: self.config['drizzle_pixfrac'] = float(config['drizzle_pixfrac'])
                # Accepter 'none' pour la normalisation
                if 'normalization' in config and config['normalization'] in ['astropy', 'skimage', 'basic', 'linear_fit', 'sky_mean', 'none']:
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
        if method in ['linear_fit', 'sky_mean']:
            if method == 'linear_fit':
                return self._normalize_images_linear_fit(images, reference_index=0)
            else:
                return self._normalize_images_sky_mean(images, reference_index=0)

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

    def _normalize_images_linear_fit(self, images, reference_index=0, low_percentile=25.0, high_percentile=90.0):
        normalized = []
        if not images:
            return []
        if not (0 <= reference_index < len(images)) or images[reference_index] is None:
            return [img.copy() if img is not None else None for img in images]

        ref = images[reference_index].astype(np.float32, copy=False)
        is_color = ref.ndim == 3 and ref.shape[-1] == 3
        num_channels = ref.shape[-1] if is_color else 1
        ref_stats = []
        for c in range(num_channels):
            ref_ch = ref[..., c] if is_color else ref
            low = np.nanpercentile(ref_ch, low_percentile)
            high = np.nanpercentile(ref_ch, high_percentile)
            ref_stats.append((low, high))
        for i, img in enumerate(images):
            if img is None:
                normalized.append(None)
                continue
            if i == reference_index:
                normalized.append(ref.copy())
                continue
            cur = img.astype(np.float32, copy=False)
            if cur.shape != ref.shape:
                normalized.append(cur.copy())
                continue
            cur = cur.copy()
            for c in range(num_channels):
                ch = cur[..., c] if is_color else cur
                low_s = np.nanpercentile(ch, low_percentile)
                high_s = np.nanpercentile(ch, high_percentile)
                ref_low, ref_high = ref_stats[c]
                a = 1.0
                b = 0.0
                d_src = high_s - low_s
                d_ref = ref_high - ref_low
                if abs(d_src) > 1e-5:
                    if abs(d_ref) > 1e-5:
                        a = d_ref / d_src
                        b = ref_low - a * low_s
                    else:
                        b = ref_low - low_s
                else:
                    if abs(d_ref) > 1e-5:
                        a = 0.0
                        b = ref_low
                    else:
                        b = ref_low - low_s
                transformed = a * ch + b
                if is_color:
                    cur[..., c] = transformed
                else:
                    cur = transformed
            normalized.append(cur)
        return normalized

    def _normalize_images_sky_mean(self, images, reference_index=0, sky_percentile=25.0):
        if not images:
            return []
        if not (0 <= reference_index < len(images)) or images[reference_index] is None:
            return [img.copy() if img is not None else None for img in images]

        ref = images[reference_index].astype(np.float32, copy=True)
        if ref.ndim == 3 and ref.shape[-1] == 3:
            luminance_ref = 0.299 * ref[...,0] + 0.587 * ref[...,1] + 0.114 * ref[...,2]
            ref_sky = np.nanpercentile(luminance_ref, sky_percentile)
        else:
            ref_sky = np.nanpercentile(ref, sky_percentile)
        out = []
        for i, img in enumerate(images):
            if img is None:
                out.append(None)
                continue
            cur = img.astype(np.float32, copy=True)
            if i == reference_index:
                out.append(cur)
                continue
            if cur.ndim == 3 and cur.shape[-1] == 3:
                luminance = 0.299 * cur[...,0] + 0.587 * cur[...,1] + 0.114 * cur[...,2]
                sky = np.nanpercentile(luminance, sky_percentile)
            else:
                sky = np.nanpercentile(cur, sky_percentile)
            if np.isfinite(sky):
                cur += (ref_sky - sky)
            out.append(cur)
        return out

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

######################################################################################################################################################

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

##################################################################################################################################################################

def apply_low_wht_mask(
    img: np.ndarray,
    wht: np.ndarray,
    *,
    percentile: float = 10.0,
    soften_px: int = 128,
    # seuil absolu en‑dessous duquel on considère qu'un poids est nul
    min_threshold: float = 1e-5,
    # fraction maximale de l'image qu'on autorise à masquer (0‑1)
    max_mask_fraction: float = 0.50,
    progress_callback=None,
) -> np.ndarray:
    """Masque les bords à faible WHT de façon robuste.

    Paramètres
    ----------
    img : ndarray (H,W,3 | H,W) float32
        Image normalisée 0‑1.
    wht : ndarray (H,W) float32
        Carte de poids normalisée (0‑max_images empilées).
    percentile : float, optionnel (1‑20)
        Percentile servant de point de coupure *initial*.
    soften_px : int, optionnel
        Rayon (px) du flou gaussien appliqué sur le masque binaire.
    min_threshold : float, optionnel
        Valeur plancher pour le *threshold* afin d'éviter un masque total.
    max_mask_fraction : float, optionnel
        Sécurité : si le masque couvrirait > *max_mask_fraction* de l'image
        le traitement est ignoré.
    progress_callback : callable | None
        Fonction de log.

    Retour
    ------
    ndarray float32
        Image 0‑1 après masquage (ou intacte si opération annulée).
    """

    def _log(msg: str):
        if callable(progress_callback):
            progress_callback(f"   [LowWHTMask] {msg}", None)
        else:
            print(f"DEBUG [LowWHTMask]: {msg}")

    # --- Vérifs ---------------------------------------------------------------
    if img is None or wht is None:
        _log("Image ou WHT manquant – masque ignoré.")
        return img
    if img.shape[:2] != wht.shape:
        _log("Dimensions img / WHT incompatibles – masque ignoré.")
        return img
    if img.ndim not in (2, 3):
        _log("Image doit être 2D (mono) ou 3D (RGB).")
        return img

    img_f32 = img.astype(np.float32, copy=True)
    wht_f32 = wht.astype(np.float32, copy=False)

    # --- 1. Seuil adaptatif ----------------------------------------------------
    positive = wht_f32[wht_f32 > min_threshold]
    if positive.size == 0:
        _log("Tous les poids sont nuls → on n'applique rien.")
        return img

    # seuil initial basé sur le percentile demandé
    thresh_initial = np.percentile(positive, percentile)
    # on ne permet pas que le seuil dépasse la médiane -> masque max 50 %
    thresh_wht = min(thresh_initial, np.median(positive))
    thresh_wht = max(thresh_wht, min_threshold)
    _log(f"Percentile : {percentile:.1f} → seuil brut {thresh_initial:.5f}, appliqué {thresh_wht:.5f}")

    # --- 2. Création / adoucissement du masque --------------------------------
    binary_mask = (wht_f32 > thresh_wht).astype(np.float32)
    masked_fraction = 1.0 - binary_mask.mean()
    _log(f"Part de l'image qui serait masquée : {masked_fraction*100:.1f}%")
    if masked_fraction > max_mask_fraction:
        _log("Masque > max_mask_fraction – opération annulée.")
        return img

    k = max(1, soften_px // 2) * 2 + 1  # noyau impair obligatoire pour cv2
    soft_mask = cv2.GaussianBlur(binary_mask, (k, k), 0)
    _log(f"Masque adouci : min={soft_mask.min():.3f} max={soft_mask.max():.3f}")

    # --- 3. Couleur de remplissage -------------------------------------------
    if img_f32.ndim == 3:
        # médiane → moins sensible aux valeurs extrêmes très chaudes
        fill_color = np.median(img_f32, axis=(0, 1)).astype(np.float32)
    else:
        fill_color = np.median(img_f32).astype(np.float32)

    _log(f"Couleur de remplissage : {fill_color}")

    # --- 4. Application -------------------------------------------------------
    if img_f32.ndim == 3:
        img_out = img_f32 * soft_mask[..., None] + fill_color[None, None, :] * (1 - soft_mask[..., None])
    else:
        img_out = img_f32 * soft_mask + fill_color * (1 - soft_mask)

    img_out = np.clip(img_out, 0.0, 1.0).astype(np.float32)

    # --- 5. Vérification dynamique -------------------------------------------
    dyn = img_out.max() - img_out.min()
    _log(f"Dynamique finale : {dyn:.4f}")
    if dyn < 0.05:
        _log("Dynamique < 0.05 → masque très destructif, on revient à l'image d'origine.")
        return img

    return img_out






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