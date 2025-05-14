# --- START OF FILE seestar/enhancement/drizzle_integration.py ---
import numpy as np
import os
import traceback
import warnings
import time
import gc
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.coordinates import SkyCoord 
from astropy import units as u
import cv2
# ConvexHull n'est pas utilisé dans ce fichier

try:
    import colour_demosaicing
    _DEBAYER_AVAILABLE = True
    print("DEBUG DrizzleIntegration: Found colour_demosaicing.")
except ImportError:
    _DEBAYER_AVAILABLE = False
    print("WARNING DrizzleIntegration: colour-demosaicing library not found.")
    class colour_demosaicing: # Factice
        @staticmethod
        def demosaicing_CFA_Bayer_Malvar2004(data, pattern):
            print("ERROR: colour_demosaicing not available for demosaicing.")
            return data

try:
    from drizzle.resample import Drizzle
    _OO_DRIZZLE_AVAILABLE = True
    print("DEBUG DrizzleIntegration: Imported drizzle.resample.Drizzle")
except ImportError:
    _OO_DRIZZLE_AVAILABLE = False
    Drizzle = None # Pour éviter NameError si Drizzle non importé
    print("ERROR DrizzleIntegration: Cannot import drizzle.resample.Drizzle class")

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
                # print(f"DEBUG _load_drizzle_temp_file: HDU invalide pour {filepath}")
                return None, None, None
            hdu = hdul[0]; data = hdu.data; header = hdu.header
            data_hxwx3 = None
            if data.ndim == 3:
                if data.shape[2] == 3: data_hxwx3 = data.astype(np.float32)
                elif data.shape[0] == 3: data_hxwx3 = np.moveaxis(data, 0, -1).astype(np.float32)
                else: 
                    # print(f"DEBUG _load_drizzle_temp_file: Shape 3D inattendue {data.shape} pour {filepath}")
                    return None, None, None
            else: 
                # print(f"DEBUG _load_drizzle_temp_file: Données non 3D {data.ndim}D pour {filepath}")
                return None, None, None
            wcs = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', FITSFixedWarning)
                    wcs_hdr = WCS(header, naxis=2)
                if wcs_hdr.is_celestial: wcs = wcs_hdr
            except Exception: pass
            if wcs is None: 
                # print(f"DEBUG _load_drizzle_temp_file: WCS non trouvé/valide pour {filepath}")
                return None, None, None
            return data_hxwx3, wcs, header
    except FileNotFoundError: 
        # print(f"DEBUG _load_drizzle_temp_file: Fichier non trouvé {filepath}")
        return None, None, None
    except Exception as e: 
        print(f"ERREUR _load_drizzle_temp_file pour {filepath}: {e}")
        traceback.print_exc(limit=1); return None, None, None

def _create_wcs_from_header(header): # Nom corrigé
    required_keys = ['NAXIS1', 'NAXIS2', 'RA', 'DEC', 'FOCALLEN', 'XPIXSZ', 'YPIXSZ']
    if not all(key in header for key in required_keys): 
        # print(f"DEBUG _create_wcs_from_header: Clés manquantes { [k for k in required_keys if k not in header] }")
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
        print(f"ERREUR _create_wcs_from_header: {e_wcs_create}")
        return None

def _load_fits_data_wcs_debayered(filepath, bayer_pattern='GRBG'): # Nom corrigé
    # print(f"DEBUG _load_fits_data_wcs_debayered: Tentative chargement {filepath}")
    try:
        with fits.open(filepath, memmap=False) as hdul:
            hdu = None; header = None
            for h_item in hdul:
                if h_item.is_image and h_item.data is not None and h_item.data.ndim == 2:
                    hdu = h_item; break
            if hdu is None: 
                # print(f"DEBUG _load_fits_data_wcs_debayered: Aucune HDU 2D image valide dans {filepath}")
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
                    print(f"ERREUR Debayer dans _load_fits_data_wcs_debayered pour {filepath}: {e_debayer}")
                    return None, None, None # Échec du debayering
            else: 
                print(f"WARN _load_fits_data_wcs_debayered: Bibliothèque colour_demosaicing non dispo pour {filepath}.")
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
                # print(f"DEBUG _load_fits_data_wcs_debayered: WCS non trouvé/généré pour {filepath}")
                return None, None, None # WCS est essentiel
            
            return rgb_image.astype(np.float32), wcs, header
            
    except FileNotFoundError: 
        # print(f"DEBUG _load_fits_data_wcs_debayered: Fichier non trouvé {filepath}")
        return None, None, None
    except Exception as e_load: 
        print(f"ERREUR _load_fits_data_wcs_debayered pour {filepath}: {e_load}")
        traceback.print_exc(limit=1); return None, None, None

# === CLASSE DrizzleProcessor ===
class DrizzleProcessor:
    """
    Encapsule la logique pour appliquer Drizzle en utilisant la classe drizzle.resample.Drizzle.
    """
    def __init__(self, scale_factor=2.0, pixfrac=1.0, kernel='square', fillval="0.0", final_wht_threshold=0.1):
        if not _OO_DRIZZLE_AVAILABLE or Drizzle is None:
            print("ERREUR DrizzleProcessor: Classe Drizzle (drizzle.resample.Drizzle) non disponible.")
            raise ImportError("Classe Drizzle (drizzle.resample.Drizzle) non disponible. Veuillez installer 'drizzle'.")

        self.scale_factor = float(max(1.0, scale_factor))
        self.pixfrac = float(np.clip(pixfrac, 0.01, 1.0))
        valid_kernels = ['square', 'gaussian', 'point', 'tophat', 'turbo', 'lanczos2', 'lanczos3']
        self.kernel = kernel.lower() if kernel.lower() in valid_kernels else 'square'
        self.fillval = str(fillval) 
        self.final_wht_threshold = float(np.clip(final_wht_threshold, 0.0, 1.0))
        
        print(f"DEBUG DrizzleProcessor Initialized (using drizzle.resample.Drizzle): "
              f"ScaleFactor(info)={self.scale_factor}, Pixfrac={self.pixfrac}, Kernel='{self.kernel}', "
              f"Fillval='{self.fillval}', FinalWHTThresh={self.final_wht_threshold}")





# --- DANS LA CLASSE DrizzleProcessor DANS seestar/enhancement/drizzle_integration.py ---

    def apply_drizzle(self, 
                      input_file_paths: list, 
                      output_wcs: WCS, 
                      output_shape_2d_hw: tuple,
                      use_local_alignment_logic: bool = False,
                      anchor_wcs_for_local: WCS = None,
                      progress_callback: callable = None
                      ):
        if not progress_callback: 
            progress_callback = lambda msg, prog=None: print(f"DrizzleProc LOG: {msg}" + (f" ({prog}%)" if prog is not None else ""))
        
        print(f"DEBUG DrizzleProcessor.apply_drizzle (V_InspectorToolFix): Appelée avec {len(input_file_paths)} fichiers.") # Log version
        # ... (vérifications initiales inchangées) ...
        if not _OO_DRIZZLE_AVAILABLE or Drizzle is None: 
            progress_callback("Drizzle ERREUR: Classe Drizzle (drizzle.resample.Drizzle) non disponible.", 0)
            return None, None
        # ... (autres vérifications)

        num_output_channels = 3 
        out_images_by_channel = [np.zeros(output_shape_2d_hw, dtype=np.float32) for _ in range(num_output_channels)]
        out_weights_by_channel = [np.zeros(output_shape_2d_hw, dtype=np.float32) for _ in range(num_output_channels)]
        drizzlers_by_channel = []

        try:
            progress_callback(f"Drizzle: Initialisation pour {num_output_channels} canaux (Shape sortie: {output_shape_2d_hw})...", None)
            for i in range(num_output_channels):
                # --- CORRECTION INITIALISATION DRIZZLE SELON VOTRE INSPECTION ---
                driz_ch = Drizzle( 
                    out_img=out_images_by_channel[i],  
                    out_wht=out_weights_by_channel[i],
                    kernel=self.kernel,   # self.kernel (de DrizzleProcessor) est passé ici
                    fillval=self.fillval  # self.fillval (de DrizzleProcessor) est passé ici
                    # pixfrac n'est PAS pour __init__
                )
                # --- FIN CORRECTION ---
                
                drizzlers_by_channel.append(driz_ch)
            progress_callback("Drizzle: Initialisation Drizzle terminée.", None)
            print(f"DEBUG DrizzleProcessor (V_InspectorToolFix): Initialisation des Drizzlers terminée.")
        except Exception as e_init_driz:
            progress_callback(f"Drizzle ERREUR: Initialisation Drizzle échouée: {e_init_driz}", 0)
            traceback.print_exc(limit=1)
            return None, None

        files_processed_count = 0
        for i, file_path in enumerate(input_file_paths):
            # ... (début de la boucle inchangé : chargement fichier, calcul pixmap/inwcs) ...
            if progress_callback and (i % 10 == 0 or i == len(input_file_paths) - 1): 
                 progress_callback(f"Drizzle: Traitement image {i+1}/{len(input_file_paths)}...", int(i / len(input_file_paths) * 100) if len(input_file_paths) > 0 else 0)
            
            input_image_cxhxw = None 
            input_header = None
            exposure_time = 1.0 

            try:
                with fits.open(file_path, memmap=False) as hdul:
                    if not hdul or len(hdul) == 0 or hdul[0].data is None: continue
                    input_image_cxhxw = hdul[0].data.astype(np.float32) 
                    input_header = hdul[0].header
                
                if input_image_cxhxw.ndim != 3 or input_image_cxhxw.shape[0] != num_output_channels: continue
                input_shape_hw = (input_image_cxhxw.shape[1], input_image_cxhxw.shape[2]) 

                if input_header and 'EXPTIME' in input_header:
                    try: exposure_time = max(1e-6, float(input_header['EXPTIME']))
                    except (ValueError, TypeError): pass
                
                current_pixmap_for_drizzle = None
                input_wcs_for_drizzle = None 

                if use_local_alignment_logic: # Calcul pixmap
                    if anchor_wcs_for_local is None: continue
                    try:
                        M_matrix = np.array([[input_header.get('M11',1.0),input_header.get('M12',0.0),input_header.get('M13',0.0)],
                                             [input_header.get('M21',0.0),input_header.get('M22',1.0),input_header.get('M23',0.0)]], dtype=np.float32)
                        if 'M11' not in input_header: M_matrix = np.array([[1.,0.,0.],[0.,1.,0.]], dtype=np.float32)
                    except KeyError: continue
                    y_orig, x_orig = np.indices(input_shape_hw)
                    pts_orig = np.dstack((x_orig.ravel(),y_orig.ravel())).astype(np.float32).reshape(-1,1,2)
                    pts_anchor = cv2.transform(pts_orig, M_matrix).reshape(-1,2)
                    sky_ra, sky_dec = anchor_wcs_for_local.all_pix2world(pts_anchor[:,0], pts_anchor[:,1], 0)
                    final_x, final_y = output_wcs.all_world2pix(sky_ra, sky_dec, 0)
                    current_pixmap_for_drizzle = np.dstack((final_x.reshape(input_shape_hw), final_y.reshape(input_shape_hw))).astype(np.float32)
                else: # Obtenir inwcs
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore"); input_wcs_for_drizzle = WCS(input_header, naxis=2)
                        if not input_wcs_for_drizzle.is_celestial: input_wcs_for_drizzle = None
                    except Exception: input_wcs_for_drizzle = None
                    if input_wcs_for_drizzle is None: continue
                
                for ch_idx in range(num_output_channels):
                    channel_data_2d = input_image_cxhxw[ch_idx, :, :].astype(np.float32)
                    if not np.all(np.isfinite(channel_data_2d)): channel_data_2d[~np.isfinite(channel_data_2d)] = 0.0
                    
                    # --- CORRECTION APPEL add_image SELON VOTRE INSPECTION ---
                    if use_local_alignment_logic:
                        if current_pixmap_for_drizzle is None: continue
                        drizzlers_by_channel[ch_idx].add_image(
                            data=channel_data_2d, 
                            pixmap=current_pixmap_for_drizzle, 
                            exptime=exposure_time,
                            in_units='cps' if self.kernel == 'turbo' else 'counts', # self.kernel de DrizzleProcessor
                            pixfrac=self.pixfrac    # self.pixfrac de DrizzleProcessor est passé ici
                            # kernel N'EST PAS un argument de add_image
                        )
                    else: 
                        if input_wcs_for_drizzle is None: continue
                        drizzlers_by_channel[ch_idx].add_image(
                            data=channel_data_2d, 
                            inwcs=input_wcs_for_drizzle, 
                            outwcs=output_wcs,
                            exptime=exposure_time,
                            in_units='cps' if self.kernel == 'turbo' else 'counts', # self.kernel de DrizzleProcessor
                            pixfrac=self.pixfrac    # self.pixfrac de DrizzleProcessor est passé ici
                        )
                    # --- FIN CORRECTION APPEL ---
                files_processed_count += 1
            except Exception as e_file_proc:
                progress_callback(f"Drizzle ERREUR: Traitement fichier {file_path} échoué: {e_file_proc}", None)
                traceback.print_exc(limit=1)
            finally:
                del input_image_cxhxw, input_header, input_wcs_for_drizzle, current_pixmap_for_drizzle
                if (i + 1) % 20 == 0: gc.collect()
        
        # ... (reste de la fonction apply_drizzle : récupération des résultats, etc. - inchangé)
        progress_callback(f"Drizzle: Boucle Drizzle terminée. {files_processed_count}/{len(input_file_paths)} fichiers traités.", 100)
        print(f"DEBUG DrizzleProcessor (V_InspectorToolFix): Boucle Drizzle terminée. {files_processed_count} fichiers effectivement ajoutés.")

        if files_processed_count == 0:
            # ...
            return None, None

        try:
            # ...
            final_sci_image_hxwxc = np.stack(out_images_by_channel, axis=-1).astype(np.float32) 
            final_wht_map_hxwxc = np.stack(out_weights_by_channel, axis=-1).astype(np.float32)
            # ... (nettoyage et application seuil wht)
            return final_sci_image_hxwxc, final_wht_map_hxwxc
        except Exception as e_final_stack:
            # ...
            return None, None
        finally:
            # ...
            gc.collect()






# --- FIN DU FICHIER seestar/enhancement/drizzle_integration.py ---