# --- START OF FILE seestar/enhancement/stack_enhancement.py (MODIFIED) ---
import cv2
import numpy as np
from skimage import exposure
import time
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
            'edge_crop_percent': 0.1   # % à rogner sur chaque bord (0 = pas de rognage)
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

    def process(self, images, use_drizzle=False, images_headers=None):
        """
        Pipeline MAJ v5: Gère Drizzle ou retourne None pour le chemin classique.
        Le stacking classique est maintenant géré par queue_manager._stack_batch.
        Cette méthode gère UNIQUEMENT le chemin Drizzle.

        Args:
            images (list): Liste d'images (numpy arrays, float32, 0-1). Déjà alignées.
            use_drizzle (bool): Utiliser Drizzle pour le stacking.
            images_headers (list, optional): Headers pour Drizzle WCS.

        Returns:
            tuple: (final_image, final_weight_map) si Drizzle réussi, ou (None, None).
        """
        # --- Initialisation et vérifications ---
        if not images:
            print("StackEnhancer: Aucune image à traiter.")
            return None, None
        images_processed = [img for img in images if img is not None and img.size > 0]
        if not images_processed:
            print("StackEnhancer: Aucune image valide fournie.")
            return None, None

        start_time = time.time()
        print(f"--- Début StackEnhancer Process (Mode: {'Drizzle' if use_drizzle else 'PostProcessing Only - via apply_postprocessing()'}) ---")

        stacked_image = None # Image résultant du stacking (Drizzle seulement)
        weight_map = None    # Carte de poids (uniquement pour Drizzle réussi)

        # --- Étape 1: Stacking (Soit Drizzle, soit RIEN ici) ---

        # --- Chemin Drizzle ---
        if use_drizzle:
            if _drizzle_available and DrizzleProcessor:
                try:
                    print(f"--> StackEnhancer: Appel DrizzleProcessor...")
                    drizzle_proc = DrizzleProcessor(
                        scale_factor=self.config['drizzle_scale'],
                        pixfrac=self.config['drizzle_pixfrac'],
                        # kernel='square' # Utilise le défaut du constructeur DrizzleProcessor
                    )
                    # Passer les headers s'ils sont fournis
                    stacked_image, weight_map = drizzle_proc.apply_drizzle(
                        images_processed, images_headers=images_headers
                    )
                    if stacked_image is None:
                        raise RuntimeError("Drizzle a échoué (apply_drizzle a retourné None).")
                    print(f"    -> Drizzle réussi (Shape: {stacked_image.shape})")

                except ImportError: # Juste au cas où
                     print("StackEnhancer: Erreur Import DrizzleProcessor.")
                     stacked_image, weight_map = None, None
                except Exception as drizzle_err:
                    print(f"   -> ERREUR pendant Drizzle: {drizzle_err}")
                    traceback.print_exc(limit=2)
                    stacked_image, weight_map = None, None
            else:
                print("   -> ERREUR CRITIQUE: Drizzle demandé mais non disponible/importé.")
                stacked_image, weight_map = None, None
        # --- Fin du chemin Drizzle ---

        # --- Chemin Stacking Classique (NE FAIT PLUS RIEN ICI) ---
        else: # if not use_drizzle:
            print("StackEnhancer.process: Mode non-Drizzle. Retourne None (stacking fait ailleurs).")
            stacked_image, weight_map = None, None
            # Pas besoin de continuer le post-traitement ici si on ne fait pas le stack
            end_time = time.time()
            print(f"--- Fin StackEnhancer Process (Durée: {end_time - start_time:.2f}s) ---")
            return None, None # Important de retourner None si pas Drizzle
        # --- Fin Chemin Stacking Classique ---


        # --- Vérification après Drizzle ---
        if stacked_image is None:
            print("StackEnhancer: Échec de l'étape de stacking Drizzle.")
            return None, None # Retourner tuple

        # --- Étape 2: Post-traitement des bords (UNIQUEMENT pour Drizzle ici) ---
        final_image_before_norm = None
        if use_drizzle and weight_map is not None: # On a réussi Drizzle
            print("... Application du masque basé sur la carte de poids Drizzle (WHT)...")
            try:
                # Utilisation du seuil WHT de la config
                wht_threshold_ratio = self.config.get('drizzle_wht_threshold', 0.7)
                max_wht = np.nanmax(weight_map)
                if max_wht > 1e-9:
                    threshold_wht = max_wht * wht_threshold_ratio
                    mask = weight_map >= threshold_wht
                    print(f"    -> Seuil WHT appliqué: {threshold_wht:.2f} ({wht_threshold_ratio*100:.0f}% du max {max_wht:.2f})")
                    if stacked_image.ndim == 3 and mask.ndim == 2:
                         mask = np.expand_dims(mask, axis=-1)
                    final_image_before_norm = np.where(mask, stacked_image, 0.0)
                else:
                     print("    -> Warning: Carte WHT vide ou nulle, pas de masquage appliqué.")
                     final_image_before_norm = stacked_image
            except Exception as wht_err:
                print(f"    -> ERREUR application masque WHT: {wht_err}")
                traceback.print_exc(limit=1)
                final_image_before_norm = stacked_image # Fallback
        else:
             # Si Drizzle a réussi mais pas de WHT, ou si on voulait du rognage aussi
             # On pourrait appliquer le rognage % ici si souhaité, mais WHT est mieux pour Drizzle
             final_image_before_norm = stacked_image

        # --- Étape 3: CLAHE (Toujours désactivé pour ce test) ---
        print("... Application CLAHE DÉSACTIVÉE pour test...")
        # final_image_clahe = self._apply_clahe(final_image_before_norm)
        # if final_image_clahe is None: print("StackEnhancer: Échec CLAHE."); return None, None
        # final_image_processed = final_image_clahe
        final_image_processed = final_image_before_norm # Utiliser l'image avant CLAHE

        # --- Étape 4: Normalisation Finale (Min-Max) ---
        print("... Normalisation finale Min-Max...")
        min_f, max_f = np.nanmin(final_image_processed), np.nanmax(final_image_processed)
        if max_f > min_f:
            final_image = (final_image_processed - min_f) / (max_f - min_f)
        else:
            final_image = np.zeros_like(final_image_processed)
        final_image = np.clip(final_image, 0.0, 1.0).astype(np.float32) # Assurer float32

        end_time = time.time()
        print(f"--- Fin StackEnhancer Process (Durée: {end_time - start_time:.2f}s) ---")

        # --- Retourner le tuple final ---
        return final_image, weight_map

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