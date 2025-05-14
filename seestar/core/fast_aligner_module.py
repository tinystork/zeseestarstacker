# --- START OF FILE seestar/core/fast_aligner_module.py ---
"""
Module pour l'alignement rapide d'images basé sur la détection de features OpenCV.
Contient FastAligner et un adaptateur SeestarAligner pour compatibilité.
"""
import cv2
import numpy as np
import traceback # Ajouté pour un meilleur logging d'erreur

print("DEBUG [FastAlignerModule]: Module en cours de chargement...")

# =============================================================================
#  FAST ALIGNER – Alignement rapide (translation + rotation + échelle)
# =============================================================================
class FastAligner:
    """Alignement inspiré de DeepSkyStacker (100 % local, sans réseau).

    ▸ Phase 1 : détection d’étoiles avec *SimpleBlobDetector* (Optionnel, ORB est souvent suffisant)
    ▸ Phase 2 : ORB + BF-Matcher + RANSAC → matrice affine 2 × 3
    ▸ Phase 3 : application de la matrice sur l’image RGB ou mono

    Toutes les actions sont *loggées* si `debug=True`.
    """

    # ------------------------------------------------------------------
    #  INITIALISATION
    # ------------------------------------------------------------------
    def __init__(self, debug: bool = False):
        self.debug = bool(debug)
        self.progress_callback = None  # Pour éventuel relais vers l’UI
        if self.debug:
            print("DEBUG [FastAligner]: Instance créée avec debug =", self.debug)

    # ------------------------------------------------------------------
    #  LOGGING / PROGRESS
    # ------------------------------------------------------------------
    def set_progress_callback(self, cb):
        """Enregistre un callback (msg:str, progress:int|None) -> None."""
        self.progress_callback = cb
        if self.debug:
            print("DEBUG [FastAligner]: Progress callback défini.")

    def _log(self, msg: str, level: str = "INFO"):
        tag = f"[FastAligner/{level}] "
        # Toujours imprimer si debug est activé
        if self.debug:
            print(tag + msg)
        
        # Appeler le callback s'il existe, même si debug est False, pour le GUI
        if callable(self.progress_callback):
            try:
                self.progress_callback(tag + msg, None) # Le GUI s'attend à deux args pour progress_callback
            except Exception as e_cb:
                if self.debug:
                    print(tag + f"(Erreur callback ignorée: {e_cb})")

    # ------------------------------------------------------------------
    #  PHASE 1 : détection d’étoiles (points brillants) - Actuellement non utilisé par estimate_transform
    # ------------------------------------------------------------------
    def _detect_stars(self, img_gray: np.ndarray,
                      blob_min_threshold: int = 10, # Augmenté un peu
                      blob_max_threshold: int = 200, # Réduit un peu
                      min_area_px: float = 5.0,    # Un peu plus grand pour éviter le bruit
                      max_area_px: float = 200.0) -> list: # Moins grand, les très grosses étoiles peuvent varier
        """Retourne une liste de cv2.KeyPoint."""
        self._log("Phase 1: Début détection des étoiles (SimpleBlobDetector)...")

        # — normalisation vers uint8 pour le détecteur
        img_f32 = img_gray.astype(np.float32, copy=False)
        # S'assurer que l'image n'est pas plate avant normalisation
        min_val, max_val = np.min(img_f32), np.max(img_f32)
        if max_val <= min_val:
            self._log("Image plate ou vide pour détection étoiles. Retourne liste vide.", level="WARN")
            return []
            
        img_norm = cv2.normalize(img_f32, None, 0.0, 1.0, cv2.NORM_MINMAX)
        img_u8 = (np.clip(img_norm, 0, 1) * 255).astype(np.uint8)
        
        if self.debug:
            self._log(f"Image pour BlobDetector normalisée → uint8 ; shape={img_u8.shape}")

        # — configuration du blob‑detector
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True # Filtrer par couleur (intensité)
        params.blobColor = 255      # Détecter les blobs blancs (lumineux)
        
        params.minThreshold = float(blob_min_threshold) # OpenCV attend des floats ici
        params.maxThreshold = float(blob_max_threshold)
        
        params.filterByArea = True
        params.minArea = float(min_area_px)
        params.maxArea = float(max_area_px)
        
        # Désactiver les autres filtres pour commencer
        params.filterByCircularity = False
        # params.minCircularity = 0.6 # Typique pour les étoiles si activé
        params.filterByConvexity = False
        # params.minConvexity = 0.85 # Typique si activé
        params.filterByInertia = False
        # params.minInertiaRatio = 0.1 # Typique si activé

        try:
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(img_u8) # Renvoie une liste de cv2.KeyPoint
        except Exception as e_blob:
            self._log(f"Erreur lors de la création ou utilisation de SimpleBlobDetector: {e_blob}", level="ERROR")
            traceback.print_exc()
            return []

        # Pas besoin de convertir en np.array ici si ORB les utilise directement
        # stars_coords = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
        
        self._log(f"Phase 1: Détection terminée. {len(keypoints)} keypoints (étoiles) trouvés par BlobDetector.")
        return keypoints # Retourner la liste de cv2.KeyPoint

    # ------------------------------------------------------------------
    #  PHASE 2 : estimation de la matrice affine relative
    # ------------------------------------------------------------------

    def estimate_transform(self, 
                           ref_gray: np.ndarray, 
                           img_gray: np.ndarray,
                           min_matches_ratio: float = 0.15, 
                           min_absolute_matches: int = 10,  
                           ransac_thresh: float = 5.0,      
                           orb_features: int = 2000):       
        """Retourne une matrice 2 × 3 ou None si échec."""
        self._log(f"Phase 2: Début estimation de la transformation (ORB)... Features: {orb_features}, MinMatchRatio: {min_matches_ratio}, MinAbsMatch: {min_absolute_matches}, RANSAC Thresh: {ransac_thresh}")

        ref_f32 = ref_gray.astype(np.float32, copy=False)
        img_f32 = img_gray.astype(np.float32, copy=False)
        
        min_r, max_r = np.min(ref_f32), np.max(ref_f32)
        if max_r <= min_r: self._log("Image de référence plate ou vide pour ORB.", level="WARN"); return None
        ref_u8 = cv2.normalize(ref_f32, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        min_i, max_i = np.min(img_f32), np.max(img_f32)
        if max_i <= min_i: self._log("Image source plate ou vide pour ORB.", level="WARN"); return None
        img_u8 = cv2.normalize(img_f32, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        try:
            orb = cv2.ORB_create(nfeatures=orb_features) 
            kp1, des1 = orb.detectAndCompute(ref_u8, None)
            kp2, des2 = orb.detectAndCompute(img_u8, None)
        except Exception as e_orb:
            self._log(f"Erreur lors de la détection/calcul ORB: {e_orb}", level="ERROR")
            traceback.print_exc()
            return None

        if des1 is None or len(kp1) == 0:
            self._log("Pas de descripteurs ORB pour l'image de RÉFÉRENCE.", level="ERROR")
            return None
        if des2 is None or len(kp2) == 0:
            self._log("Pas de descripteurs ORB pour l'image SOURCE.", level="ERROR")
            return None
            
        self._log(f"ORB: {len(kp1)} keypoints sur Réf, {len(kp2)} keypoints sur Src.")

        try:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            # matcher.match() retourne une liste de DMatch
            raw_matches = matcher.match(des1, des2) 
            print(f"DEBUG [FastAligner]: Type de raw_matches APRES matcher.match: {type(raw_matches)}") # <-- AJOUT DEBUG
            if raw_matches: # Si ce n'est pas None ou une liste vide
                 print(f"DEBUG [FastAligner]: Nombre de raw_matches: {len(raw_matches)}")
                 if len(raw_matches) > 0:
                     print(f"DEBUG [FastAligner]: Type du premier élément de raw_matches: {type(raw_matches[0])}")

        except Exception as e_match:
            self._log(f"Erreur pendant le matching BFMatcher: {e_match}", level="ERROR")
            traceback.print_exc()
            return None

        if not raw_matches: # raw_matches peut être une liste vide
            self._log("Aucune correspondance brute trouvée par BFMatcher.", level="ERROR")
            return None
        
        # Si raw_matches est un tuple (ce qui serait étrange), le convertir en liste avant de trier
        if isinstance(raw_matches, tuple):
            self._log("raw_matches était un tuple, conversion en liste.", level="WARN")
            raw_matches = list(raw_matches)
            
        # S'assurer que raw_matches est bien une liste avant d'appeler .sort()
        if not isinstance(raw_matches, list):
            self._log(f"raw_matches n'est pas une liste (type: {type(raw_matches)}), impossible de trier. Matching échoué.", level="ERROR")
            return None

        # Vérifier que les éléments sont bien des DMatch et ont l'attribut distance
        if raw_matches and not all(hasattr(m, 'distance') for m in raw_matches):
            self._log("Tous les éléments de raw_matches n'ont pas d'attribut 'distance'. Matching échoué.", level="ERROR")
            return None

        try:
            raw_matches.sort(key=lambda m: m.distance) # Tri en place de la liste
        except AttributeError as e_sort: # Si un élément n'a pas 'distance'
            self._log(f"Erreur de tri sur raw_matches (AttributeError: {e_sort}). Un élément n'a peut-être pas 'distance'.", level="ERROR")
            traceback.print_exc()
            return None
        except Exception as e_sort_other: # Autres erreurs de tri
            self._log(f"Erreur de tri inconnue sur raw_matches: {e_sort_other}", level="ERROR")
            traceback.print_exc()
            return None

        num_good_matches_to_keep = max(min_absolute_matches, int(len(raw_matches) * min_matches_ratio))
        good_matches = raw_matches[:num_good_matches_to_keep]
        
        if len(good_matches) < min_absolute_matches:
            self._log(f"Pas assez de 'bonnes' correspondances ({len(good_matches)} < {min_absolute_matches} requis) après filtrage initial.", level="ERROR")
            return None
        
        self._log(f"{len(raw_matches)} correspondances brutes -> {len(good_matches)} 'bonnes' correspondances sélectionnées pour RANSAC.")

        pts1_ref = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2_src = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        try:
            M, inliers_mask = cv2.estimateAffinePartial2D(pts2_src, pts1_ref, 
                                                         method=cv2.RANSAC,
                                                         ransacReprojThreshold=ransac_thresh,
                                                         maxIters=5000, 
                                                         confidence=0.995)
        except Exception as e_ransac:
            self._log(f"Erreur pendant cv2.estimateAffinePartial2D (RANSAC): {e_ransac}", level="ERROR")
            traceback.print_exc()
            return None
                                                         
        if M is None:
            self._log("RANSAC n’a pas trouvé de matrice de transformation valide.", level="ERROR")
            return None
        
        num_inliers = np.sum(inliers_mask)
        self._log(f"Matrice de transformation trouvée. Inliers: {num_inliers} / {len(good_matches)}.")
        
        min_ransac_inliers = max(6, min_absolute_matches // 2)
        if num_inliers < min_ransac_inliers:
            self._log(f"Nombre d'inliers RANSAC ({num_inliers}) trop faible (requis: {min_ransac_inliers}). Transformation rejetée.", level="WARN")
            return None

        if self.debug:
            self._log(f"Matrice M calculée :\n{M}")
        return M






    # ------------------------------------------------------------------
    #  PHASE 3 : warp de l’image (tous canaux)
    # ------------------------------------------------------------------
    def warp_image(self, 
                   img_to_warp: np.ndarray, 
                   M: np.ndarray, 
                   output_shape_wh: tuple, # Doit être (width, height) pour dsize OpenCV
                   border_mode=cv2.BORDER_CONSTANT, # Ou BORDER_REFLECT, BORDER_REPLICATE
                   border_value=(0,0,0,0)): # Valeur pour BORDER_CONSTANT (noir par défaut)
        """Applique M via `cv2.warpAffine` et renvoie l’image alignée."""
        if M is None:
            self._log("Matrice de transformation est None, impossible de warper.", level="ERROR")
            # Retourner l'image originale non modifiée au lieu de lever une erreur pour permettre au flux de continuer
            return img_to_warp 
        
        self._log(f"Phase 3: Application de la transformation (warp). Shape de sortie cible (W,H): {output_shape_wh}")
        try:
            warped_image = cv2.warpAffine(img_to_warp, M, 
                                          dsize=output_shape_wh, # OpenCV attend (width, height)
                                          flags=cv2.INTER_LINEAR, # INTER_LANCZOS4 peut être mieux mais plus lent
                                          borderMode=border_mode,
                                          borderValue=border_value)
            return warped_image
        except Exception as e_warp:
            self._log(f"Erreur lors de cv2.warpAffine: {e_warp}", level="ERROR")
            traceback.print_exc()
            return img_to_warp # Retourner l'original en cas d'erreur de warp

# =============================================================================
#  ADAPTATEUR : interface similaire à astroalign ou l'ancien SeestarAligner
# =============================================================================
class FastSeestarAligner: # Nom changé pour éviter confusion avec celui dans core.alignment
    """
    Enrobe `FastAligner` pour fournir une interface `_align_image`
    similaire à ce qui pourrait être attendu, retournant (aligned_image, success_boolean).
    """

    def __init__(self, debug: bool = False):
        self._fa = FastAligner(debug=debug) # Instance de notre FastAligner
        self.set_progress_callback(None) # Initialiser
        if debug:
            print("DEBUG [FastSeestarAligner]: Instance créée.")

    def set_progress_callback(self, cb):
        """Passe le callback à l'instance FastAligner interne."""
        self._fa.set_progress_callback(cb)
        if self._fa.debug:
            print("DEBUG [FastSeestarAligner]: Progress callback transmis à FastAligner interne.")






# DANS LA CLASSE FastSeestarAligner DANS seestar/core/fast_aligner_module.py

    def _align_image(self, 
                     src_img: np.ndarray,
                     ref_img: np.ndarray,
                     file_name: str | None = None) -> tuple[np.ndarray | None, np.ndarray | None, bool]: # MODIFIÉ: Retourne M
        """
        Tente d'aligner src_img sur ref_img.

        Args:
            src_img (np.ndarray): Image source (peut être HxW ou HxWxC, float 0-1).
            ref_img (np.ndarray): Image de référence (peut être HxW ou HxWxC, float 0-1).
            file_name (str, optional): Nom de fichier pour logging.

        Returns:
            tuple: (aligned_image_float32_0_1, M_transform_matrix, success_boolean)
                   aligned_image et M_transform_matrix sont None si l'alignement échoue.
        """
        tag = file_name or "[image sans nom]"
        self._fa._log(f"Début alignement pour '{tag}' avec FastSeestarAligner...")

        src_f32 = src_img.astype(np.float32, copy=False)
        ref_f32 = ref_img.astype(np.float32, copy=False)

        def to_gray_if_color(im_f32: np.ndarray) -> np.ndarray:
            # ... (logique inchangée) ...
            if im_f32.ndim == 2:
                return im_f32
            elif im_f32.ndim == 3 and im_f32.shape[2] == 3: # RGB
                return cv2.cvtColor(im_f32, cv2.COLOR_RGB2GRAY)
            elif im_f32.ndim == 3 and im_f32.shape[2] == 4: # RGBA
                return cv2.cvtColor(im_f32, cv2.COLOR_RGBA2GRAY)
            else: 
                self._fa._log(f"Format d'image inattendu pour conversion gris: {im_f32.shape}. Tentative avec premier canal.", level="WARN")
                if im_f32.ndim ==3: return im_f32[..., 0]
                return im_f32

        src_gray_f32 = to_gray_if_color(src_f32)
        ref_gray_f32 = to_gray_if_color(ref_f32)

        M = self._fa.estimate_transform(ref_gray_f32, src_gray_f32, 
                                        orb_features=2500, 
                                        ransac_thresh=5.0, 
                                        min_absolute_matches=8) 
        
        if M is None:
            self._fa._log(f"Alignement ÉCHOUÉ pour '{tag}': estimate_transform a retourné None.", level="ERROR")
            return None, None, False # MODIFIÉ: Retourne M=None

        output_h, output_w = ref_f32.shape[:2]
        output_shape_wh_cv = (output_w, output_h)

        aligned_image = None
        if src_f32.ndim == 2: 
            aligned_image = self._fa.warp_image(src_f32, M, output_shape_wh_cv)
        elif src_f32.ndim == 3 and src_f32.shape[2] in [3, 4]: 
            channels = cv2.split(src_f32)
            warped_channels = []
            num_channels_to_warp = src_f32.shape[2]
            for i in range(num_channels_to_warp):
                warped_channel = self._fa.warp_image(channels[i], M, output_shape_wh_cv)
                warped_channels.append(warped_channel)
            aligned_image = cv2.merge(warped_channels)
        else:
            self._fa._log(f"Format d'image source non supporté pour warp: {src_f32.shape}", level="ERROR")
            return None, M, False # MODIFIÉ: Retourne M même si warp échoue pour info

        if aligned_image is img_f32: # Si warp_image a retourné l'original en cas d'erreur interne
             self._fa._log(f"Alignement PARTIELLEMENT ÉCHOUÉ pour '{tag}': warp_image n'a pas modifié l'image.", level="WARN")
             return None, M, False # MODIFIÉ: Retourne M mais échec

        aligned_image = np.clip(aligned_image.astype(np.float32), 0.0, 1.0)

        self._fa._log(f"Alignement RÉUSSI pour '{tag}'. Shape de sortie: {aligned_image.shape}")
        return aligned_image, M, True # MODIFIÉ: Retourne M











print("DEBUG [FastAlignerModule]: Module chargé avec succès.")
# --- END OF FILE seestar/core/fast_aligner_module.py ---