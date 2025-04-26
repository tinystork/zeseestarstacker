# --- START OF FILE seestar/enhancement/drizzle_integration.py (CORRECTED STRUCTURE) ---
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

if not _standalone_drizzle_available:
    def dummy_tdriz(*args, **kwargs):
        raise ImportError("Standalone drizzle function (tdriz) not available.")
    tdriz_function = dummy_tdriz

warnings.filterwarnings('ignore', category=FITSFixedWarning)

# --- DEBUT DE LA CLASSE (UNE SEULE FOIS) ---
class DrizzleProcessor:
    def __init__(self, scale_factor=2.0, pixfrac=1.0, kernel='square'):
        """
        Initialise le processeur Drizzle utilisant la fonction drizzle.cdrizzle.tdriz.
        """
        if not _standalone_drizzle_available:
            raise ImportError("Standalone drizzle library (tdriz) not found or import failed.")
        # Validation
        if not isinstance(scale_factor, (int, float)) or scale_factor < 1: scale_factor = 2.0
        if not isinstance(pixfrac, (int, float)) or not (0.0 < pixfrac <= 1.0): pixfrac = 1.0
        valid_kernels = ['square', 'gaussian', 'point', 'tophat', 'turbo', 'lanczos2', 'lanczos3']
        if kernel not in valid_kernels: kernel = 'square'
        self.scale_factor = float(scale_factor)
        self.pixfrac = float(pixfrac)
        self.kernel = kernel
        print(f"DrizzleProcessor (tdriz) Initialized: ScaleFactor={self.scale_factor}, Pixfrac={self.pixfrac}, Kernel='{self.kernel}'")

    # --- MÉTHODE _create_output_wcs (BIEN INDENTÉE) ---
    def _create_output_wcs(self, input_shape, input_header=None):
        """ Crée WCS de sortie, en s'assurant que la matrice PC est définie. """
        h, w = input_shape[:2]
        out_h = int(round(h * self.scale_factor))
        out_w = int(round(w * self.scale_factor))
        crpix1, crpix2 = w / 2.0 + 0.5, h / 2.0 + 0.5
        crval1, crval2 = 0.0, 0.0
        ctype1, ctype2 = "RA---TAN", "DEC--TAN"
        cdelt1_deg, cdelt2_deg = -(1.2 / 3600.0), (1.2 / 3600.0) # Default Seestar
        valid_wcs_from_header = False
        if input_header:
            try:
                wcs_in = WCS(input_header)
                if wcs_in.is_celestial and wcs_in.has_celestial:
                    crval1, crval2 = wcs_in.wcs.crval
                    try:
                        cd11 = wcs_in.wcs.cd[0, 0]; cd12 = wcs_in.wcs.cd[0, 1]
                        cd21 = wcs_in.wcs.cd[1, 0]; cd22 = wcs_in.wcs.cd[1, 1]
                        cdelt1_deg = np.sign(cd11) * np.sqrt(cd11**2 + cd21**2)
                        cdelt2_deg = np.sign(cd22) * np.sqrt(cd12**2 + cd22**2)
                        # print(f"DEBUG (WCS Output): CD/PC matrix found.") # Moins de logs
                    except AttributeError:
                        if wcs_in.wcs.cdelt is not None:
                             cdelt1_deg, cdelt2_deg = wcs_in.wcs.cdelt
                             # print(f"DEBUG (WCS Output): Using CDELT from header.")
                    crpix1 = float(input_header.get('CRPIX1', crpix1))
                    crpix2 = float(input_header.get('CRPIX2', crpix2))
                    ctype1 = input_header.get('CTYPE1', ctype1)
                    ctype2 = input_header.get('CTYPE2', ctype2)
                    valid_wcs_from_header = True
                else: raise ValueError("WCS non céleste")
            except Exception as wcs_e:
                 print(f"DEBUG (WCS Output): Header WCS invalid ({wcs_e}). Using defaults.")
        out_wcs = WCS(naxis=2)
        out_wcs.wcs.crpix = [out_w / 2.0 + 0.5, out_h / 2.0 + 0.5]
        out_wcs.wcs.crval = [crval1, crval2]; out_wcs.wcs.ctype = [ctype1, ctype2]
        out_wcs.wcs.cdelt = np.array([cdelt1_deg / self.scale_factor, cdelt2_deg / self.scale_factor])
        out_wcs.wcs.pc = np.array([[1.0, 0.0], [0.0, 1.0]])
        out_wcs.pixel_shape = (out_w, out_h)
        return out_wcs

    # --- MÉTHODE _create_input_wcs_basic (BIEN INDENTÉE) ---
    def _create_input_wcs_basic(self, input_shape, input_header=None):
        """
        Crée un WCS d'entrée basique. Tente d'utiliser le header fourni,
        mais si cela échoue, crée un WCS minimaliste complètement de novo.
        """
        h, w = input_shape[:2]
        if input_header:
            try:
                with warnings.catch_warnings():
                     warnings.simplefilter('ignore', FITSFixedWarning)
                     wcs_from_header = WCS(input_header)
                if wcs_from_header.is_celestial and wcs_from_header.has_celestial:
                    if not hasattr(wcs_from_header.wcs, 'pc') or wcs_from_header.wcs.pc is None:
                         wcs_from_header.wcs.pc = np.array([[1.0, 0.0], [0.0, 1.0]])
                    return wcs_from_header
                else: pass # Fallback
            except Exception as e: pass # Fallback

        print("DEBUG (WCS Input): Création WCS de novo minimaliste.")
        try:
            min_wcs = WCS(naxis=2)
            min_wcs.wcs.crpix = [w / 2.0 + 0.5, h / 2.0 + 0.5]
            min_wcs.wcs.crval = [0.0, 0.0]; min_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            min_wcs.wcs.cdelt = np.array([-(1.2 / 3600.0), (1.2 / 3600.0)])
            min_wcs.wcs.pc = np.array([[1.0, 0.0], [0.0, 1.0]])
            min_wcs.wcs.cunit = ['deg', 'deg']
            if min_wcs.is_celestial: return min_wcs
            else: print("ERREUR WCS de novo non céleste!"); return None
        except Exception as e_min: print(f"ERREUR création WCS de novo: {e_min}"); return None

    # --- MÉTHODE apply_drizzle (BIEN INDENTÉE et version de test simplifiée) ---
    def apply_drizzle(self, filepath_list, images_headers=None):
        """
        Applique Drizzle en utilisant tdriz sur une liste de fichiers FITS temporaires.
        Tente un appel simplifié à tdriz pour débogage.
        """
        start_time = time.time()
        if not _standalone_drizzle_available: return None, None
        if not filepath_list: return None, None

        print(f"DrizzleProcessor (tdriz): Application Drizzle sur {len(filepath_list)} fichiers temporaires (APPEL SIMPLIFIÉ POUR TEST).")
        output_wcs = None; final_sci = None; final_wht = None
        is_color = False; ref_shape = None

        try:
            num_images = len(filepath_list)
            print(f"  -> Lecture fichiers et application tdriz séquentielle...")
            processed_count = 0

            for i, filepath in enumerate(filepath_list):
                if i == 0 or (i + 1) % 50 == 0 or i == num_images - 1:
                    print(f"     ... traitement fichier {i+1}/{num_images}: {os.path.basename(filepath)}")

                try:
                    with fits.open(filepath, memmap=False) as hdul:
                        if not hdul or hdul[0].data is None or hdul[0].data.size == 0:
                            print(f"     ⚠️ Fichier vide ou invalide: {os.path.basename(filepath)}"); continue
                        img_data = hdul[0].data.astype(np.float32); header = hdul[0].header
                        if processed_count == 0:
                            current_shape = img_data.shape
                            is_color = img_data.ndim == 3 and current_shape[2] == 3
                            ref_shape = current_shape
                            print(f"     -> Format détecté: {'Couleur' if is_color else 'N&B'}, Shape: {ref_shape}")
                            output_wcs = self._create_output_wcs(ref_shape, header)
                            out_h, out_w = output_wcs.pixel_shape[::-1]; out_dims = (out_h, out_w)
                            print(f"     -> Dimensions sortie: {out_dims}")
                            sci_shape = (out_h, out_w, 3) if is_color else out_dims
                            final_sci = np.zeros(sci_shape, dtype=np.float32)
                            final_wht = np.zeros(sci_shape, dtype=np.float32) # Initialiser WHT même si non utilisé par appel simplifié

                        if img_data.ndim != len(ref_shape):
                            print(f"     ⚠️ Dim. incompatibles. Ignoré."); continue
                        input_wcs = None
                        try:
                            input_wcs = WCS(header)
                            if not input_wcs.is_celestial: raise ValueError("WCS non céleste")
                        except Exception as wcs_err:
                            print(f"     ⚠️ Erreur WCS ('{wcs_err}'). Utilisation WCS basique.")
                            input_wcs = self._create_input_wcs_basic(img_data.shape, header)
                        if input_wcs is None: print(f"     ❌ Impossible créer WCS. Ignoré."); continue

                        # --- Créer les masques même pour l'appel simplifié ---
                        context_mask = np.ones(img_data.shape[:2], dtype=np.int32)
                        input_weights = np.ones(img_data.shape[:2], dtype=np.float32)

                        # --- Appel tdriz SIMPLIFIÉ (positionnels seulement) ---
                        if is_color:
                            # On doit quand même passer les bons tableaux pour les sorties
                            # même si on ne traite qu'un canal pour le test.
                            # Il vaut mieux essayer l'appel complet directement.
                            print("     DEBUG: Tentative tdriz complet (couleur)...")
                            for c in range(3):
                                channel_context_mask = np.ones(img_data[..., c].shape, dtype=np.int32)
                                channel_input_weights = np.ones(img_data[..., c].shape, dtype=np.float32)
                                tdriz_function(
                                    img_data[..., c], input_wcs, channel_input_weights, output_wcs,
                                    final_sci[..., c], 'counts', channel_context_mask,
                                    outwht=final_wht[..., c], expin=1.0, # Test avec expin=1.0
                                    pixfrac=1.0, kernel='square', # Test avec valeurs simples
                                    fillval='INDEF', wt_scl='exptime' # Essayer exptime
                                )

                        else: # N&B
                            print("     DEBUG: Tentative tdriz complet (N&B)...")
                            tdriz_function(
                                img_data, input_wcs, input_weights, output_wcs,
                                final_sci, 'counts', context_mask,
                                outwht=final_wht, expin=1.0,
                                pixfrac=1.0, kernel='square',
                                fillval='INDEF', wt_scl='exptime'
                            )
                        processed_count += 1

                except FileNotFoundError: print(f"     ❌ Fichier non trouvé: {os.path.basename(filepath)}")
                except TypeError as te_call: # Attraper spécifiquement TypeError sur l'appel
                    print(f"     ❌ TypeError lors de l'appel à tdriz pour fichier {i+1}: {te_call}")
                    traceback.print_exc(limit=2) # Montrer où dans l'appel ça échoue
                except Exception as file_err: print(f"     ❌ Erreur traitement fichier {i+1}: {file_err}"); traceback.print_exc(limit=1)

            # --- FIN BOUCLE ---

            # --- Vérification et Finalisation ---
            if processed_count == 0 or final_sci is None or final_wht is None:
                 print("❌ DrizzleProcessor: Aucune image valide traitée."); return None, None
            print(f"  -> Finalisation Drizzle ({processed_count} images)...")
            final_sci = np.nan_to_num(final_sci, nan=0.0)
            if is_color: weight_divisor = np.mean(final_wht, axis=2, dtype=np.float32); divisor_3d = np.repeat(weight_divisor[:, :, np.newaxis], 3, axis=2)
            else: weight_divisor = final_wht; divisor_3d = weight_divisor
            mask = weight_divisor > 1e-9; final_image_normalized = np.zeros_like(final_sci, dtype=np.float32)
            if is_color:
                mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                safe_divisor = np.where(divisor_3d[mask_3d] > 1e-9, divisor_3d[mask_3d], 1.0)
                final_image_normalized[mask_3d] = final_sci[mask_3d] / safe_divisor
            else:
                safe_divisor = np.where(weight_divisor[mask] > 1e-9, weight_divisor[mask], 1.0)
                final_image_normalized[mask] = final_sci[mask] / safe_divisor
            final_image_normalized = np.clip(final_image_normalized, 0.0, 1.0)
            final_wht = final_wht.astype(np.float32)
            end_time = time.time(); print(f"✅ DrizzleProcessor (tdriz): Terminé en {end_time - start_time:.2f}s.")
            return final_image_normalized, final_wht
        except ImportError: print("ERREUR Drizzle: Lib tdriz non trouvée."); return None, None
        except ValueError as val_err: print(f"ERREUR Drizzle (ValueError): {val_err}"); traceback.print_exc(limit=1); return None, None
        except Exception as e: print(f"ERREUR Drizzle Inattendue: {e}"); traceback.print_exc(limit=3); return None, None
        finally: gc.collect()

# --- FIN DE LA CLASSE ---