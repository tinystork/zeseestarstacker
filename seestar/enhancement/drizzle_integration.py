# --- START OF FILE seestar/enhancement/drizzle_integration.py (USING drizzle.cdrizzle.tdriz) ---
import numpy as np
import os
import traceback
import warnings
import time
import gc

# --- Astropy imports ---
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.coordinates import SkyCoord
from astropy import units as u

# --- Import the underlying 'tdriz' function ---
_standalone_drizzle_available = False
tdriz_function = None # Placeholder

try:
    # Importer le module cdrizzle qui contient tdriz
    from drizzle import cdrizzle
    # Vérifier si tdriz existe
    if hasattr(cdrizzle, 'tdriz') and callable(getattr(cdrizzle, 'tdriz')):
        tdriz_function = cdrizzle.tdriz # Assigner la fonction
        print("DEBUG DrizzleIntegration: Successfully found 'drizzle.cdrizzle.tdriz' function.")
        _standalone_drizzle_available = True
    else:
        raise ImportError("Function 'tdriz' not found within drizzle.cdrizzle module.")

except ImportError as e_imp:
    print(f"ERREUR CRITIQUE: Impossible d'importer ou trouver 'drizzle.cdrizzle.tdriz'. Erreur: {e_imp}")
    print("Vérifiez l'installation: pip install drizzle")
    traceback.print_exc(limit=1)

except Exception as e_other:
    print(f"ERREUR INATTENDUE pendant l'import de drizzle/cdrizzle: {type(e_other).__name__}: {e_other}")
    traceback.print_exc(limit=3)

# Créer une fonction dummy si l'import a échoué
if not _standalone_drizzle_available:
    def dummy_tdriz(*args, **kwargs):
        raise ImportError("Standalone drizzle function (tdriz) not available.")
    tdriz_function = dummy_tdriz


# Ignore FITSFixedWarning from WCS creation if headers are minimal
warnings.filterwarnings('ignore', category=FITSFixedWarning)

class DrizzleProcessor:
    def __init__(self, scale_factor=2.0, pixfrac=1.0, kernel='square'):
        """
        Initialise le processeur Drizzle utilisant la fonction drizzle.cdrizzle.tdriz.
        """
        if not _standalone_drizzle_available:
            raise ImportError("Standalone drizzle library (tdriz) not found or import failed. Cannot initialize DrizzleProcessor.")
        # Validation inchangée
        if not isinstance(scale_factor, (int, float)) or scale_factor < 1:
            print(f"Warning: Invalid scale_factor ({scale_factor}). Using 2.0.")
            scale_factor = 2.0
        if not isinstance(pixfrac, (int, float)) or not (0.0 < pixfrac <= 1.0):
             print(f"Warning: Invalid pixfrac ({pixfrac}). Using 1.0.")
             pixfrac = 1.0
        valid_kernels = ['square', 'gaussian', 'point', 'tophat', 'turbo', 'lanczos2', 'lanczos3']
        if kernel not in valid_kernels:
             print(f"Warning: Invalid kernel '{kernel}'. Using 'square'.")
             kernel = 'square'
        self.scale_factor = float(scale_factor)
        self.pixfrac = float(pixfrac)
        self.kernel = kernel
        print(f"DrizzleProcessor (tdriz) Initialized: ScaleFactor={self.scale_factor}, Pixfrac={self.pixfrac}, Kernel='{self.kernel}'")


    # --- Méthodes _create_output_wcs et _create_input_wcs_basic INCHANGÉES ---
    def _create_output_wcs(self, input_shape, input_header=None):
        """ Crée WCS de sortie (inchangé) """
        h, w = input_shape[:2]; out_h = int(round(h * self.scale_factor)); out_w = int(round(w * self.scale_factor))
        in_pix_scale_deg = (1.2 / 3600.0); cdelt1, cdelt2 = -in_pix_scale_deg, in_pix_scale_deg
        if input_header:
            try:
                if 'CDELT1' in input_header and 'CDELT2' in input_header: cdelt1,cdelt2 = float(input_header['CDELT1']), float(input_header['CDELT2'])
                elif 'CD1_1' in input_header and 'CD2_2' in input_header: cdelt1,cdelt2 = float(input_header['CD1_1']), float(input_header['CD2_2'])
                elif 'PIXSCALE' in input_header: scale_asec=float(input_header['PIXSCALE']); cdelt1,cdelt2 = -(scale_asec/3600.0), (scale_asec/3600.0)
            except Exception: pass
        out_wcs = WCS(naxis=2); out_wcs.wcs.crpix = [out_w / 2.0 + 0.5, out_h / 2.0 + 0.5]; out_wcs.wcs.crval = [0.0, 0.0]
        out_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]; out_wcs.wcs.cdelt = np.array([cdelt1 / self.scale_factor, cdelt2 / self.scale_factor])
        out_wcs.pixel_shape = (out_w, out_h); return out_wcs

    def _create_input_wcs_basic(self, input_shape, input_header=None):
        """ Crée WCS d'entrée basique (inchangé) """
        h, w = input_shape[:2]; in_pix_scale_deg = (1.2 / 3600.0); cdelt1, cdelt2 = -in_pix_scale_deg, in_pix_scale_deg
        if input_header:
            try:
                if 'CDELT1' in input_header and 'CDELT2' in input_header: cdelt1,cdelt2 = float(input_header['CDELT1']), float(input_header['CDELT2'])
                elif 'CD1_1' in input_header and 'CD2_2' in input_header: cdelt1,cdelt2 = float(input_header['CD1_1']), float(input_header['CD2_2'])
                elif 'PIXSCALE' in input_header: scale_asec=float(input_header['PIXSCALE']); cdelt1,cdelt2 = -(scale_asec/3600.0), (scale_asec/3600.0)
            except Exception: pass
        in_wcs = WCS(naxis=2); in_wcs.wcs.crpix = [w / 2.0 + 0.5, h / 2.0 + 0.5]; in_wcs.wcs.crval = [0.0, 0.0]
        in_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]; in_wcs.wcs.cdelt = np.array([cdelt1, cdelt2]); return in_wcs


    def apply_drizzle(self, images_data, images_headers=None):
        """
        Intégration Drizzle via la fonction drizzle.cdrizzle.tdriz.
        """
        start_time = time.time()
        if not _standalone_drizzle_available:
            print("DrizzleProcessor: Standalone drizzle library (tdriz) not available.")
            return None, None
        if not images_data:
            print("DrizzleProcessor: Aucune image fournie.")
            return None, None

        try:
            valid_images = [img.astype(np.float32) for img in images_data if img is not None]
            valid_headers = None
            if images_headers:
                valid_headers = [hdr for img, hdr in zip(images_data, images_headers) if img is not None]
            if not valid_images:
                print("DrizzleProcessor: Aucune image valide après filtrage.")
                return None, None

            print(f"DrizzleProcessor (tdriz): Application Drizzle sur {len(valid_images)} images.")
            ref_image = valid_images[0]
            ref_header = valid_headers[0] if valid_headers else None
            is_color = ref_image.ndim == 3 and ref_image.shape[2] == 3

            output_wcs = self._create_output_wcs(ref_image.shape, ref_header)
            out_h, out_w = output_wcs.pixel_shape
            out_h, out_w = out_w, out_h # Reorder numpy
            out_dims = (out_h, out_w)

            input_wcs_base = self._create_input_wcs_basic(ref_image.shape, ref_header)

            # --- Initialisation des tableaux de sortie ---
            # IMPORTANT: tdriz modifie ces tableaux en place !
            final_sci = np.zeros((out_h, out_w, 3) if is_color else out_dims, dtype=np.float32)
            final_wht = np.zeros((out_h, out_w, 3) if is_color else out_dims, dtype=np.float32)

            # --- Boucle Drizzle ---
            num_images = len(valid_images)
            print("  -> Application de tdriz sur les images...")
            for i, (img_data, hdr) in enumerate(zip(valid_images, valid_headers if valid_headers else [None]*num_images)):
                if i % 20 == 0: print(f"     ... image {i+1}/{num_images}")
                exptime = 1.0
                if hdr and 'EXPTIME' in hdr:
                     try: exptime = max(1.0, float(hdr['EXPTIME']))
                     except Exception: pass

                # --- Appel à tdriz_function ---
                # tdriz prend beaucoup d'arguments. Nous utilisons les essentiels.
                # Elle modifie outsci et outwht directement.
                # Il faut passer les tableaux de sortie pour *chaque* image.
                if is_color:
                    # Traiter chaque canal
                    for c in range(3):
                        # Créer des tableaux temporaires pour ce canal d'image ? Non, tdriz accumule.
                        tdriz_function(
                            img_data[..., c],       # 1. input data (2D)
                            input_wcs_base,         # 2. input WCS
                            output_wcs,             # 3. output WCS
                            final_sci[..., c],      # 4. output SCI array (positionnel)
                            outwht=final_wht[..., c],# outwht (keyword)
                            expin=exptime,
                            pixfrac=self.pixfrac,
                            kernel=self.kernel,
                            fillval='INDEF',
                            wt_scl='expsq'
                        )
                else: # N&B
                    tdriz_function(
                        img_data,               # 1. input data (2D)
                        input_wcs_base,         # 2. input WCS
                        output_wcs,             # 3. output WCS
                        final_sci,              # 4. output SCI array (positionnel)
                        outwht=final_wht,       # outwht (keyword)
                        expin=exptime,
                        pixfrac=self.pixfrac,
                        kernel=self.kernel,
                        fillval='INDEF',
                        wt_scl='expsq'
                    )

            # --- Finalisation (Normalisation, identique) ---
            print("  -> Finalisation (normalisation)...")
            final_sci = np.nan_to_num(final_sci)
            weight_divisor = final_wht[..., 1] if is_color else final_wht
            mask = weight_divisor > 1e-6
            final_image_normalized = np.zeros_like(final_sci)
            if is_color:
                for c in range(3):
                     mask_c = final_wht[..., c] > 1e-6
                     # Éviter division par zéro même si mask_c est True
                     divisor_c = final_wht[..., c][mask_c]
                     safe_divisor_c = np.where(divisor_c > 1e-9, divisor_c, 1.0) # Remplacer ~0 par 1
                     final_image_normalized[..., c][mask_c] = final_sci[..., c][mask_c] / safe_divisor_c
            else:
                 safe_divisor = np.where(weight_divisor[mask] > 1e-9, weight_divisor[mask], 1.0)
                 final_image_normalized[mask] = final_sci[mask] / safe_divisor

            final_image_normalized = np.clip(final_image_normalized, 0.0, 1.0)
            final_image_normalized = final_image_normalized.astype(np.float32)
            final_wht = final_wht.astype(np.float32)

            end_time = time.time()
            print(f"DrizzleProcessor (tdriz): Drizzle terminé en {end_time - start_time:.2f}s.")
            return final_image_normalized, final_wht

        # --- Gestion Erreurs (inchangée) ---
        except ImportError: print("ERREUR Drizzle: Bibliothèque standalone (tdriz) non trouvée lors de l'exécution."); return None, None
        except ValueError as val_err: print(f"ERREUR Drizzle (ValueError): {val_err}"); traceback.print_exc(limit=1); return None, None
        except Exception as e: print(f"ERREUR Drizzle: Erreur inattendue: {e}"); traceback.print_exc(limit=3); return None, None
        finally: gc.collect()

# --- END OF FILE seestar/enhancement/drizzle_integration.py (USING drizzle.cdrizzle.tdriz) ---