

# --- DANS LA CLASSE SeestarQueuedStacker DANS seestar/queuep/queue_manager.py ---

    def _worker(self):
        """
        Thread principal pour le traitement des images.
        Version: V5.3 (Clarified Loops, Robust Finally, Full Param Passing)
        MODIFIED: Gestion correcte du retour (tuple) de load_and_validate_fits.
        """
        print("\n" + "=" * 10 + f" DEBUG QM [_worker V5.3 - TupleFix]: Initialisation du worker " + "=" * 10)

        # --- 0.A Initialisation des attributs de session du worker ---
        self.processing_active = True
        self.processing_error = None
        start_time_session = time.monotonic()
        
        reference_image_data_for_global_alignment = None
        reference_header_for_global_alignment = None
        mosaic_ref_panel_image_data = None 
        mosaic_ref_panel_header = None     
        
        current_batch_items_with_masks_for_stack_batch = [] 
        self.intermediate_drizzle_batch_files = []          
        all_aligned_files_with_info_for_mosaic = []         

        # --- 0.B Détermination du mode d'opération ---
        use_local_aligner_for_this_mosaic_run = (
            self.is_mosaic_run and
            self.mosaic_alignment_mode in ["local_fast_fallback", "local_fast_only"] and 
            _LOCAL_ALIGNER_AVAILABLE and 
            self.local_aligner_instance is not None
        )
        use_wcs_fallback_if_local_fails = (
            use_local_aligner_for_this_mosaic_run and 
            self.mosaic_alignment_mode == "local_fast_fallback"
        )
        use_astrometry_per_panel_mosaic = (
            self.is_mosaic_run and
            self.mosaic_alignment_mode == "astrometry_per_panel"
        )
        
        print(f"DEBUG QM [_worker V5.3 - TupleFix]: Configuration de la session:")
        print(f"  - is_mosaic_run: {self.is_mosaic_run}")
        if self.is_mosaic_run:
            print(f"    - mosaic_alignment_mode: '{self.mosaic_alignment_mode}'")
            print(f"    - -> Utilisation Aligneur Local (FastAligner): {use_local_aligner_for_this_mosaic_run}")
            if use_local_aligner_for_this_mosaic_run:
                print(f"      - Fallback WCS si FastAligner échoue: {use_wcs_fallback_if_local_fails}")
            print(f"    - -> Utilisation Astrometry par Panneau: {use_astrometry_per_panel_mosaic}")
        print(f"  - drizzle_active_session (pour stacking standard non-mosaïque): {self.drizzle_active_session}")
        if self.drizzle_active_session : print(f"    - drizzle_mode (standard): '{self.drizzle_mode}'")


        try:
            # === SECTION 1: PRÉPARATION DE L'IMAGE DE RÉFÉRENCE ET DU WCS DE RÉFÉRENCE ===
            self.update_progress("⭐ Préparation image(s) de référence...")
            if not self.current_folder or not os.path.isdir(self.current_folder): 
                raise RuntimeError(f"Dossier d'entrée initial invalide : {self.current_folder}")
            
            initial_files_in_first_folder = sorted([f for f in os.listdir(self.current_folder) if f.lower().endswith((".fit", ".fits"))])
            if not initial_files_in_first_folder and not self.additional_folders: # Si pas de fichiers initiaux ET pas d'autres dossiers à scanner
                raise RuntimeError("Aucun fichier FITS initial trouvé dans le premier dossier et aucun dossier additionnel en attente.")

            # Configurer l'aligneur pour _get_reference_image
            self.aligner.correct_hot_pixels = self.correct_hot_pixels
            self.aligner.hot_pixel_threshold = self.hot_pixel_threshold
            self.aligner.neighborhood_size = self.neighborhood_size
            self.aligner.bayer_pattern = self.bayer_pattern
            # self.aligner.reference_image_path est déjà setté par start_processing
            
            # Obtenir l'image de référence qui sera utilisée pour l'alignement global (astroalign)
            # OU comme base pour le premier panneau de la mosaïque locale.
            print(f"DEBUG QM [_worker V5.3 - TupleFix]: Appel à self.aligner._get_reference_image pour la référence de base/globale...")
            reference_image_data_for_global_alignment, reference_header_for_global_alignment = self.aligner._get_reference_image(
                self.current_folder, initial_files_in_first_folder
            )
            if reference_image_data_for_global_alignment is None or reference_header_for_global_alignment is None:
                # _get_reference_image devrait déjà avoir loggué l'erreur
                raise RuntimeError("Échec critique obtention image/header de référence de base (globale/premier panneau).")
            
            # Stocker une copie du header de référence pour la génération WCS (sera mis à jour)
            self.reference_header_for_wcs = reference_header_for_global_alignment.copy()
            # Ajouter le nom du fichier source de la référence au header WCS pour traçabilité
            if reference_header_for_global_alignment.get('_SOURCE_PATH'): # _SOURCE_PATH est ajouté par _get_reference_image
                self.reference_header_for_wcs['_REFSRCFN'] = (
                    os.path.basename(str(reference_header_for_global_alignment.get('_SOURCE_PATH',''))), 
                    "Base name of global ref source"
                )
            
            # Sauvegarder une copie de cette image de référence (visuelle et FITS)
            self.aligner._save_reference_image(
                reference_image_data_for_global_alignment, 
                reference_header_for_global_alignment, 
                self.output_folder
            )
            print(f"DEBUG QM [_worker V5.3 - TupleFix]: Image de référence de base (globale/premier panneau) prête. Shape: {reference_image_data_for_global_alignment.shape}")

            # --- 1.A Plate-solving du panneau de référence (SI mode local) OU de la référence globale ---
            self.reference_wcs_object = None # Sera le WCS SOLVED de la référence (ancrage ou globale)
            
            solve_image_wcs_func = None # Importer tardivement pour éviter dépendances circulaires au chargement
            try: 
                from ..enhancement.astrometry_solver import solve_image_wcs as siw_f
                solve_image_wcs_func = siw_f
                print("DEBUG QM [_worker V5.3 - TupleFix]: Import dynamique solve_image_wcs RÉUSSI.")
            except ImportError as e_siw: 
                print(f"ERREUR QM [_worker V5.3 - TupleFix]: Import dynamique solve_image_wcs ÉCHOUÉ: {e_siw}")
                # Si astrometry_solver est critique, on pourrait raise ici.
                # Pour l'instant, on continue et on gère si solve_image_wcs_func reste None.

            if use_local_aligner_for_this_mosaic_run:
                self.update_progress("⭐ Mosaïque Locale: Traitement du panneau de référence (ancrage)...")
                # L'image de référence globale devient notre panneau de référence pour la mosaïque locale
                mosaic_ref_panel_image_data = reference_image_data_for_global_alignment 
                mosaic_ref_panel_header = reference_header_for_global_alignment.copy() # Utiliser une copie
                # Ajouter une clé spécifique pour marquer l'origine de ce panneau de référence
                if reference_header_for_global_alignment.get('_SOURCE_PATH'):
                    mosaic_ref_panel_header['_PANREF_FN'] = (
                        os.path.basename(str(reference_header_for_global_alignment.get('_SOURCE_PATH',''))), 
                        "Base name of this mosaic ref panel source"
                    )
                
                temp_wcs_ancre = None
                if solve_image_wcs_func:
                    self.update_progress("   -> Mosaïque Locale: Résolution astrométrique du panneau de référence (ancrage)...")
                    temp_wcs_ancre = solve_image_wcs_func(
                        mosaic_ref_panel_image_data, 
                        mosaic_ref_panel_header, 
                        self.api_key,
                        scale_est_arcsec_per_pix=self.reference_pixel_scale_arcsec, 
                        progress_callback=self.update_progress
                    )
                else: # solve_image_wcs_func n'a pas pu être importé
                    self.update_progress("   ⚠️ Mosaïque Locale: solve_image_wcs non disponible. Tentative WCS approximatif pour panneau réf...")

                if temp_wcs_ancre is None: # Fallback si Astrometry.net échoue ou n'est pas disponible
                    self.update_progress("   ⚠️ Échec Astrometry.net pour panneau de référence. Tentative WCS approximatif...", None)
                    _cwfh_func = None # Import tardif
                    try: 
                        from ..enhancement.drizzle_integration import _create_wcs_from_header as _cwfh
                        _cwfh_func = _cwfh
                        print("DEBUG QM [_worker V5.3 - TupleFix]: Import dynamique _create_wcs_from_header RÉUSSI.")
                    except ImportError as e_cwfh:
                         print(f"ERREUR QM [_worker V5.3 - TupleFix]: Import dynamique _create_wcs_from_header ÉCHOUÉ: {e_cwfh}")
                    
                    if _cwfh_func: 
                        temp_wcs_ancre = _cwfh_func(mosaic_ref_panel_header) 
                        if temp_wcs_ancre and temp_wcs_ancre.is_celestial:
                            # Essayer d'attacher pixel_shape si possible (crucial pour certaines opérations WCS)
                            nx_hdr = mosaic_ref_panel_header.get('NAXIS1', None)
                            ny_hdr = mosaic_ref_panel_header.get('NAXIS2', None)
                            if nx_hdr and ny_hdr:
                                try: temp_wcs_ancre.pixel_shape = (int(nx_hdr), int(ny_hdr))
                                except ValueError: print("   WARN QM: NAXIS1/2 non entiers pour pixel_shape WCS fallback.")
                            elif hasattr(mosaic_ref_panel_image_data, 'shape'): # Fallback sur la shape des données image
                                temp_wcs_ancre.pixel_shape = (mosaic_ref_panel_image_data.shape[1], mosaic_ref_panel_image_data.shape[0]) # (W, H)
                            if temp_wcs_ancre.pixel_shape: self.update_progress("   -> WCS approximatif panneau réf. OK (avec pixel_shape).", None)
                            else: self.update_progress("   -> WCS approximatif panneau réf. OK (SANS pixel_shape).", None)
                        else: self.update_progress("   -> Échec création WCS approximatif panneau réf.", None)
                    else: self.update_progress("   -> Fonction _create_wcs_from_header non disponible pour fallback WCS panneau réf.", None)

                if temp_wcs_ancre is None: 
                    raise RuntimeError("Mosaïque Locale: Échec critique obtention WCS (Astrometry ET fallback) pour panneau de référence.")
                
                # Le WCS d'ancrage (résolu ou fallback) devient notre WCS de référence pour cette session
                self.reference_wcs_object = temp_wcs_ancre
                
                # Ajouter le panneau de référence (avec son WCS absolu et une matrice identité) à la liste pour la mosaïque
                mat_identite_ref_panel = np.array([[1.,0.,0.],[0.,1.,0.]], dtype=np.float32) # Matrice identité car c'est la référence
                valid_mask_ref_panel_pixels = np.ones(mosaic_ref_panel_image_data.shape[:2], dtype=bool) # Tout est valide
                
                all_aligned_files_with_info_for_mosaic.append(
                    (mosaic_ref_panel_image_data.copy(),        # Données image pré-traitées du panneau réf.
                     mosaic_ref_panel_header.copy(),            # Son header (avec _PANREF_FN)
                     self.reference_wcs_object,                 # Le WCS absolu (résolu ou fallback) de ce panneau
                     mat_identite_ref_panel,                    # Sa transformation par rapport à lui-même (identité)
                     valid_mask_ref_panel_pixels)               # Masque de pixels valides
                )
                self.aligned_files_count += 1 # Le panneau de référence compte comme une image alignée
                self.processed_files_count += 1 # Et aussi comme une image traitée
                print(f"DEBUG QM [_worker V5.3 - TupleFix]: Mosaïque Locale: Panneau de référence (image originale pré-traitée) ajouté à la liste de mosaïque.")
                wcs_type_log_ref = "Astrometry.net" if hasattr(self.reference_wcs_object, 'sip') and self.reference_wcs_object.sip is not None else "Fallback/Approximatif"
                print(f"  -> WCS du panneau de référence (ancrage) est de type: {wcs_type_log_ref}")

            elif self.drizzle_active_session or use_astrometry_per_panel_mosaic:
                # Pour Drizzle standard (non-mosaïque) ou Mosaïque avec Astrometry par panneau,
                # nous avons besoin d'un WCS précis pour l'image de référence globale.
                self.update_progress("   -> Résolution astrométrique de la référence principale (pour Drizzle standard / Mosaïque Astrometry)...")
                if solve_image_wcs_func:
                    self.reference_wcs_object = solve_image_wcs_func(
                        reference_image_data_for_global_alignment, 
                        self.reference_header_for_wcs, # Utiliser le header global
                        self.api_key,
                        scale_est_arcsec_per_pix=self.reference_pixel_scale_arcsec, 
                        progress_callback=self.update_progress
                    )
                else: # solve_image_wcs_func n'a pas pu être importé
                    self.update_progress("   ⚠️ solve_image_wcs non disponible pour référence principale. Traitement Drizzle/Mosaïque Astrometry risque d'échouer.")
                
                if self.reference_wcs_object is None: 
                    # Si Drizzle ou Mosaïque Astrometry est actif, c'est une erreur critique.
                    if self.drizzle_active_session or use_astrometry_per_panel_mosaic:
                        raise RuntimeError("Échec plate-solving de la référence principale (requis pour Drizzle/Mosaïque Astrometry).")
                    else: # Pas critique si on ne fait que du stacking classique sans Drizzle/Mosaïque
                        self.update_progress("   -> INFO: Plate-solving de la référence principale a échoué, mais pas requis pour le mode actuel.")
                else:
                    print(f"DEBUG QM [_worker V5.3 - TupleFix]: WCS de référence principale (pour Drizzle/Mosaïque Astrometry) obtenu.")
            
            self.update_progress("⭐ Référence(s) prête(s).", 5)
            self._recalculate_total_batches() # Mettre à jour estimation des lots
            self.update_progress(f"▶️ Démarrage boucle principale (En file: {self.files_in_queue} | Lots Estimés: {self.total_batches_estimated if self.total_batches_estimated > 0 else '?'})...")

            # Garder une trace du nom de fichier du panneau de référence local pour le skipper dans la boucle
            path_of_processed_ref_panel_basename = None
            if use_local_aligner_for_this_mosaic_run and mosaic_ref_panel_header and '_PANREF_FN' in mosaic_ref_panel_header:
                path_of_processed_ref_panel_basename = mosaic_ref_panel_header['_PANREF_FN']
            
            # === BOUCLE 2: BOUCLE PRINCIPALE DE TRAITEMENT DE LA FILE D'ATTENTE ===
            while not self.stop_processing:
                # --- 2.A Initialisation des variables pour cette itération de la boucle ---
                file_path = None # Chemin du fichier FITS en cours de traitement
                # Variables pour les résultats du traitement d'un fichier (utilisées par toutes les branches)
                aligned_data_item=None; header_item=None; quality_scores_item=None; wcs_object_indiv_item=None; valid_pixel_mask_item=None
                
                # Variables spécifiques à la branche Mosaïque Locale
                # current_panel_data_loaded=None; # Sera initialisé après load_and_validate_fits
                # current_panel_header=None;      # Sera initialisé après load_and_validate_fits ou getheader
                current_panel_data_processed_for_align=None # Données prêtes pour l'aligneur local
                _aligned_img_temp_from_local_aligner=None; M_transform_from_local_aligner=None; align_success_local_aligner=False    
                wcs_approx_panel_for_fallback=None; M_matrix_from_wcs_fallback=None # Pour le fallback WCS
                
                try: # TRY Principal de la boucle (pour self.queue.get et le traitement de fichier/lot)
                    file_path = self.queue.get(timeout=1.0) 
                    file_name_for_log = os.path.basename(file_path)
                    print(f"DEBUG QM [_worker V5.3 - TupleFix / Boucle Principale]: Traitement fichier '{file_name_for_log}' depuis la queue.")
                    
                    # --- TRY Secondaire: Traitement effectif du fichier ---
                    try: 
                        # --- BRANCHE 2.B.1: Traitement Mosaïque Locale (FastAligner) ---
                        if use_local_aligner_for_this_mosaic_run:
                            print(f"  DEBUG QM [_worker / Mosaïque Locale]: Début traitement pour '{file_name_for_log}'.")
                            # Vérifier si le fichier courant est le panneau de référence déjà traité
                            if path_of_processed_ref_panel_basename and file_name_for_log == path_of_processed_ref_panel_basename:
                                print(f"  DEBUG QM [_worker / Mosaïque Locale]: Fichier '{file_name_for_log}' est le panneau de référence local (déjà traité et ajouté). Consommé et skippé.")
                                # Le compteur processed_files_count a déjà été incrémenté pour le panneau de réf.
                                # On met path_of_processed_ref_panel_basename à None pour ne pas skipper d'autres fichiers
                                path_of_processed_ref_panel_basename = None 
                            else:
                                self.update_progress(f"   -> Mosaïque Locale: Alignement local de '{file_name_for_log}'...")
                                
                                ### MODIFICATION START: Gestion retour load_and_validate_fits ###
                                print(f"    DEBUG QM [_worker / Mosaïque Locale]: Appel load_and_validate_fits pour panneau '{file_name_for_log}'...")
                                loaded_panel_data_tuple = load_and_validate_fits(file_path) # Renvoie (data, header)
                                
                                current_panel_data_loaded_from_fits = None
                                current_panel_header_from_fits = None

                                if loaded_panel_data_tuple is not None and loaded_panel_data_tuple[0] is not None:
                                    current_panel_data_loaded_from_fits, current_panel_header_from_fits = loaded_panel_data_tuple
                                    print(f"    DEBUG QM [_worker / Mosaïque Locale]: load_and_validate_fits OK pour '{file_name_for_log}'. Shape data: {current_panel_data_loaded_from_fits.shape}")
                                else: # Échec chargement
                                    self.update_progress(f"   -> Mosaïque Locale: Échec chargement/validation FITS pour panneau '{file_name_for_log}'. Ignoré.")
                                    print(f"    ERREUR QM [_worker / Mosaïque Locale]: Échec chargement via load_and_validate_fits pour '{file_name_for_log}'.")
                                    self.processed_files_count += 1 
                                    self.failed_align_count +=1 # Ou un autre compteur d'erreur
                                    raise ValueError(f"Échec critique chargement FITS panneau local '{file_name_for_log}'") # Levera une exception gérée par le bloc externe
                                
                                # S'assurer d'avoir un header pour le panneau
                                panel_header_for_processing = None
                                if current_panel_header_from_fits is not None:
                                    panel_header_for_processing = current_panel_header_from_fits.copy()
                                else: # Fallback si header non retourné (ne devrait pas arriver si données OK)
                                    print(f"    WARN QM [_worker / Mosaïque Locale]: Header non retourné par load_and_validate pour '{file_name_for_log}'. Tentative getheader.")
                                    try: panel_header_for_processing = fits.getheader(file_path)
                                    except Exception as e_gethdr_panel: 
                                        self.update_progress(f"   -> Mosaïque Locale: Échec fatal lecture header pour '{file_name_for_log}': {e_gethdr_panel}. Ignoré.")
                                        raise ValueError(f"Échec critique lecture header panneau local '{file_name_for_log}': {e_gethdr_panel}")
                                
                                panel_header_for_processing['_SRCFILE'] = (file_name_for_log, "Original source file name for this panel")
                                
                                # current_panel_data_loaded_from_fits contient les données image (np.array) normalisées 0-1 float32
                                current_panel_data_processed_for_align = current_panel_data_loaded_from_fits.astype(np.float32) # Déjà float32, mais copy=True par défaut
                                ### MODIFICATION END ###
                                
                                # Pré-traitement du panneau courant (Debayer, HP)
                                if current_panel_data_processed_for_align.ndim == 2: # Si monochrome, tenter debayering
                                    bayer_pat_panel = panel_header_for_processing.get('BAYERPAT', self.bayer_pattern)
                                    pattern_upper_panel = bayer_pat_panel.upper() if isinstance(bayer_pat_panel, str) else self.bayer_pattern.upper()
                                    if pattern_upper_panel in ["GRBG", "RGGB", "GBRG", "BGGR"]:
                                        print(f"    DEBUG QM [_worker / Mosaïque Locale]: Debayering panneau '{file_name_for_log}' (Pattern: {pattern_upper_panel})...")
                                        try:
                                            current_panel_data_processed_for_align = debayer_image(current_panel_data_processed_for_align, pattern_upper_panel)
                                        except ValueError as de_panel_err:
                                            self.update_progress(f"     ⚠️ Mosaïque Locale: Erreur Debayer panneau '{file_name_for_log}' ({de_panel_err}). Traitement N&B.")
                                            print(f"    WARN QM [_worker / Mosaïque Locale]: Erreur Debayer panneau '{file_name_for_log}' (conservé N&B): {de_panel_err}")
                                
                                if self.correct_hot_pixels:
                                    print(f"    DEBUG QM [_worker / Mosaïque Locale]: Correction HP panneau '{file_name_for_log}'...")
                                    try:
                                        current_panel_data_processed_for_align = detect_and_correct_hot_pixels(
                                            current_panel_data_processed_for_align, self.hot_pixel_threshold, self.neighborhood_size
                                        )
                                    except Exception as hp_panel_err:
                                        self.update_progress(f"     ⚠️ Mosaïque Locale: Erreur correction HP panneau '{file_name_for_log}': {hp_panel_err}")
                                        print(f"    WARN QM [_worker / Mosaïque Locale]: Erreur HP panneau '{file_name_for_log}': {hp_panel_err}")
                                
                                # Alignement local avec FastSeestarAligner
                                # Il attend l'image source à aligner et l'image de référence (le panneau d'ancrage)
                                print(f"    DEBUG QM [_worker / Mosaïque Locale]: Appel à self.local_aligner_instance._align_image pour '{file_name_for_log}'...")
                                _aligned_img_temp_from_local_aligner, M_transform_from_local_aligner, align_success_local_aligner = \
                                    self.local_aligner_instance._align_image(
                                        current_panel_data_processed_for_align, # Image source (ce panneau)
                                        mosaic_ref_panel_image_data,            # Image de référence (panneau d'ancrage)
                                        file_name_for_log,                      # Nom du fichier pour logs internes à l'aligneur
                                        orb_features=self.fa_orb_features,
                                        min_absolute_matches=self.fa_min_abs_matches,
                                        min_ransac_inliers_value=self.fa_min_ransac_raw, 
                                        ransac_thresh=self.fa_ransac_thresh,
                                        min_matches_ratio=0.15 # Peut être rendu configurable via self.fa_min_matches_ratio
                                    )
                                
                                # Fallback WCS si l'alignement local a échoué ET que le mode le permet
                                if not align_success_local_aligner and use_wcs_fallback_if_local_fails:
                                    self.update_progress(f"   -> Mosaïque Locale: FastAligner échec pour '{file_name_for_log}'. Tentative fallback WCS...", None)
                                    print(f"    DEBUG QM [_worker / Mosaïque Locale Fallback]: Tentative fallback WCS pour '{file_name_for_log}'.")
                                    _cwfh_func_fallback=None # Import tardif
                                    try: 
                                        from ..enhancement.drizzle_integration import _create_wcs_from_header as _cwfh_fb
                                        _cwfh_func_fallback=_cwfh_fb
                                    except ImportError: pass
                                    
                                    if _cwfh_func_fallback:
                                        wcs_approx_panel_for_fallback = _cwfh_func_fallback(panel_header_for_processing) # Utiliser le header du panneau courant
                                        if wcs_approx_panel_for_fallback and wcs_approx_panel_for_fallback.is_celestial and self.reference_wcs_object:
                                            # Attacher pixel_shape au WCS approximatif (important pour _calculate_M_from_wcs)
                                            nx_hdr_fb = panel_header_for_processing.get('NAXIS1', None)
                                            ny_hdr_fb = panel_header_for_processing.get('NAXIS2', None)
                                            if nx_hdr_fb and ny_hdr_fb:
                                                try: wcs_approx_panel_for_fallback.pixel_shape = (int(nx_hdr_fb), int(ny_hdr_fb))
                                                except ValueError: print("     WARN QM Fallback: NAXIS1/2 non entiers pour pixel_shape WCS approx.")
                                            elif hasattr(current_panel_data_processed_for_align, 'shape'):
                                                wcs_approx_panel_for_fallback.pixel_shape = (current_panel_data_processed_for_align.shape[1], current_panel_data_processed_for_align.shape[0])
                                            
                                            if wcs_approx_panel_for_fallback.pixel_shape:
                                                print(f"    DEBUG QM [_worker / Mosaïque Locale Fallback]: WCS approximatif pour '{file_name_for_log}' OK (avec pixel_shape). Calcul de M...")
                                                M_matrix_from_wcs_fallback = self._calculate_M_from_wcs(
                                                    wcs_source=wcs_approx_panel_for_fallback,           # WCS (approx) de ce panneau
                                                    wcs_target=self.reference_wcs_object,             # WCS (précis/ancrage) du panneau de référence
                                                    shape_source_hw=current_panel_data_processed_for_align.shape[:2], # Shape de ce panneau
                                                    num_points_edge=getattr(self, 'fa_fallback_grid_pts', 4), # Peut être configurable
                                                    ransac_thresh_fallback=getattr(self, 'fa_fallback_ransac_thresh', 4.0) # Peut être configurable
                                                )
                                                if M_matrix_from_wcs_fallback is not None:
                                                    M_transform_from_local_aligner = M_matrix_from_wcs_fallback # Remplacer la matrice M
                                                    align_success_local_aligner = True # Marquer comme succès
                                                    self.update_progress(f"   -> Mosaïque Locale: Fallback WCS pour '{file_name_for_log}' RÉUSSI. Matrice M obtenue.", None)
                                                    print(f"    DEBUG QM [_worker / Mosaïque Locale Fallback]: Matrice M de fallback WCS obtenue pour '{file_name_for_log}'.")
                                                else:
                                                    self.update_progress(f"   -> Mosaïque Locale: Fallback WCS pour '{file_name_for_log}' ÉCHOUÉ (calcul M).", None)
                                                    print(f"    WARN QM [_worker / Mosaïque Locale Fallback]: Calcul M depuis WCS fallback a échoué pour '{file_name_for_log}'.")
                                            else:
                                                self.update_progress(f"   -> Mosaïque Locale: Fallback WCS pour '{file_name_for_log}' ÉCHOUÉ (pixel_shape manquant sur WCS approx).", None)
                                                print(f"    WARN QM [_worker / Mosaïque Locale Fallback]: WCS approximatif pour '{file_name_for_log}' n'a pas de pixel_shape.")
                                        else:
                                            self.update_progress(f"   -> Mosaïque Locale: Fallback WCS pour '{file_name_for_log}' ÉCHOUÉ (WCS approx invalide ou WCS réf. manquant).", None)
                                            print(f"    WARN QM [_worker / Mosaïque Locale Fallback]: WCS approx invalide ou WCS réf global manquant pour '{file_name_for_log}'.")
                                    else: # _cwfh_func_fallback non disponible
                                        self.update_progress(f"   -> Mosaïque Locale: Fallback WCS pour '{file_name_for_log}' ÉCHOUÉ (fonction _create_wcs_from_header non dispo).", None)
                                        print(f"    WARN QM [_worker / Mosaïque Locale Fallback]: _create_wcs_from_header non disponible pour fallback.")
                                
                                # Incrémenter le compteur des fichiers traités (même si alignement échoue)
                                self.processed_files_count += 1
                                if align_success_local_aligner and M_transform_from_local_aligner is not None:
                                    self.aligned_files_count += 1
                                    # Stocker: (données_originales_pré-traitées, header_original, WCS_absolu_DU_REF_PANEL, Matrice_M_vers_panneau_ref, masque=tout_valide)
                                    valid_mask_this_panel_pixels = np.ones(current_panel_data_processed_for_align.shape[:2], dtype=bool) # L'image originale est entièrement valide
                                    all_aligned_files_with_info_for_mosaic.append(
                                        (current_panel_data_processed_for_align.copy(), # Image originale (pré-traitée) de ce panneau
                                         panel_header_for_processing.copy(),            # Son header
                                         self.reference_wcs_object,                     # Le WCS de référence (ancrage) auquel M s'applique
                                         M_transform_from_local_aligner.copy(),         # La matrice M pour transformer ce panneau vers le référentiel de l'ancrage
                                         valid_mask_this_panel_pixels)                  # Son masque de validité
                                    )
                                    self.update_progress(f"   -> Mosaïque Locale: Alignement local de '{file_name_for_log}' RÉUSSI. Ajouté à la mosaïque.", None)
                                    print(f"  DEBUG QM [_worker / Mosaïque Locale]: Panneau '{file_name_for_log}' aligné localement (M stockée) et ajouté.")
                                else: # Échec final de l'alignement local (même après fallback)
                                    self.failed_align_count +=1
                                    self.update_progress(f"   -> Mosaïque Locale: Échec final alignement local pour '{file_name_for_log}'. Ignoré pour mosaïque.", None)
                                    print(f"  WARN QM [_worker / Mosaïque Locale]: Échec final alignement local pour '{file_name_for_log}'. Panneau ignoré.")
                                    # Optionnel: Déplacer le fichier vers un dossier "unaligned_mosaic_panels"
                                    if self.unaligned_folder: # Utiliser le dossier unaligned standard pour l'instant
                                        try: shutil.move(file_path, os.path.join(self.unaligned_folder, f"unaligned_MOSAIC_LOCAL_{file_name_for_log}"))
                                        except Exception: pass
                        
                        # --- BRANCHE 2.B.2: Mosaïque Astrometry par Panneau ---
                        elif use_astrometry_per_panel_mosaic:
                            print(f"  DEBUG QM [_worker / Mosaïque Astrometry]: Début traitement pour '{file_name_for_log}'.")
                            # _process_file gère le chargement, pré-traitement, et l'alignement (astroalign) sur reference_image_data_for_global_alignment.
                            # Pour la mosaïque Astrometry, on a besoin du WCS individuel résolu de chaque panneau.
                            # _process_file devrait donc être configuré pour tenter de résoudre chaque image.
                            # CEPENDANT, _process_file n'est pas conçu pour faire du plate-solving individuel.
                            # Il aligne sur une référence.
                            # Il faudrait une logique dédiée ici pour:
                            # 1. Charger, pré-traiter l'image.
                            # 2. Appeler solve_image_wcs sur cette image pré-traitée.
                            # 3. Si succès, stocker (image_pretraitee, header, wcs_resolu, masque_valide).
                            # Pour l'instant, on va supposer que _process_file est adapté pour retourner le WCS
                            # individuel s'il est généré/utilisé, OU que l'on fait le plate-solve ici.
                            #
                            # === Logique Temporaire pour Mosaïque Astrometry (à affiner) ===
                            #   Idéalement, on aurait une fonction _process_panel_for_astrometry_mosaic(file_path)
                            print(f"    DEBUG QM [_worker / Mosaïque Astrometry]: Appel _process_file pour '{file_name_for_log}'.")
                            aligned_data_item, header_item, quality_scores_item, wcs_object_indiv_item, valid_pixel_mask_item = \
                                self._process_file(file_path, reference_image_data_for_global_alignment) # La réf globale est juste pour la forme ici
                            
                            self.processed_files_count += 1
                            if aligned_data_item is not None and valid_pixel_mask_item is not None:
                                # Tenter de résoudre le WCS pour CE PANNEAU (aligned_data_item)
                                wcs_for_this_panel_astrometry = None
                                if solve_image_wcs_func:
                                    self.update_progress(f"   -> Mosaïque Astrometry: Résolution WCS pour '{file_name_for_log}'...", None)
                                    wcs_for_this_panel_astrometry = solve_image_wcs_func(
                                        aligned_data_item, # Utiliser l'image déjà pré-traitée par _process_file
                                        header_item, 
                                        self.api_key, 
                                        scale_est_arcsec_per_pix=self.reference_pixel_scale_arcsec, 
                                        progress_callback=self.update_progress
                                    )
                                
                                if wcs_for_this_panel_astrometry:
                                    self.aligned_files_count += 1
                                    # Stocker: (image_pretraitee, header, wcs_resolu_DU_PANNEAU, masque_valide)
                                    # Note: le dernier item dans le tuple est le masque de pixels valides.
                                    # Pour la mosaïque Astrometry, on n'a pas de matrice M, le WCS est absolu.
                                    all_aligned_files_with_info_for_mosaic.append(
                                        (aligned_data_item, header_item, wcs_for_this_panel_astrometry, valid_pixel_mask_item)
                                    ) # Format: (img_data_HWC, header, wcs_obj, valid_mask_2D)
                                    self.update_progress(f"   -> Mosaïque Astrometry: WCS résolu pour '{file_name_for_log}'. Ajouté.", None)
                                    print(f"  DEBUG QM [_worker / Mosaïque Astrometry]: Panneau '{file_name_for_log}' résolu et ajouté.")
                                else: # Échec plate-solve pour ce panneau
                                    self.failed_align_count +=1
                                    self.update_progress(f"   -> Mosaïque Astrometry: Échec résolution WCS pour '{file_name_for_log}'. Ignoré.", None)
                                    print(f"  WARN QM [_worker / Mosaïque Astrometry]: Échec résolution WCS pour '{file_name_for_log}'.")
                                    if self.unaligned_folder:
                                        try: shutil.move(file_path, os.path.join(self.unaligned_folder, f"unaligned_MOSAIC_ASTRO_{file_name_for_log}"))
                                        except Exception: pass
                            else: # Échec _process_file (chargement/pré-traitement)
                                self.failed_align_count +=1
                                self.update_progress(f"   -> Mosaïque Astrometry: Échec traitement initial (load/preproc) de '{file_name_for_log}'. Ignoré.", None)
                                print(f"  WARN QM [_worker / Mosaïque Astrometry]: Échec _process_file pour '{file_name_for_log}'.")
                        
                        # --- BRANCHE 2.B.3: Stacking Classique ou Drizzle Standard (non-mosaïque) ---
                        else: 
                            print(f"  DEBUG QM [_worker / Stacking Standard]: Appel _process_file pour '{file_name_for_log}'.")
                            aligned_data_item, header_item, quality_scores_item, wcs_object_indiv_item, valid_pixel_mask_item = \
                                self._process_file(file_path, reference_image_data_for_global_alignment)
                            
                            self.processed_files_count += 1
                            if aligned_data_item is not None and valid_pixel_mask_item is not None:
                                self.aligned_files_count += 1
                                # Ajouter à la liste du lot courant pour stacking classique ou Drizzle standard par lot
                                current_batch_items_with_masks_for_stack_batch.append(
                                    (aligned_data_item, header_item, quality_scores_item, wcs_object_indiv_item, valid_pixel_mask_item)
                                )
                                print(f"  DEBUG QM [_worker / Stacking Standard]: Item '{file_name_for_log}' ajouté au lot. Taille lot: {len(current_batch_items_with_masks_for_stack_batch)}.")
                                
                                # Si le lot est plein, le traiter
                                if len(current_batch_items_with_masks_for_stack_batch) >= self.batch_size:
                                    print(f"  DEBUG QM [_worker / Stacking Standard]: Lot plein ({len(current_batch_items_with_masks_for_stack_batch)}). Traitement lot...")
                                    if self.drizzle_active_session: # Drizzle Standard par lot
                                        # Préparer les données pour _process_and_save_drizzle_batch
                                        # Il attend une liste de tuples (aligned_data_HxWxC, header, wcs_object_DE_REFERENCE)
                                        # wcs_object_indiv_item de _process_file est le WCS généré pour l'image, pas forcément le WCS de référence
                                        # On doit utiliser self.reference_wcs_object (qui est le WCS résolu de l'image de référence globale)
                                        batch_data_for_drizzle_std = []
                                        for item_driz_std in current_batch_items_with_masks_for_stack_batch:
                                            img_d, hdr_d, _scores_d, _wcs_gen_d, _mask_d = item_driz_std
                                            if img_d is not None:
                                                # S'assurer qu'on a le WCS de référence global pour le Drizzle standard
                                                if self.reference_wcs_object is None and self.drizzle_active_session :
                                                    # Ceci est une erreur critique si on est en mode Drizzle standard
                                                    raise RuntimeError("WCS de référence global manquant pour Drizzle standard par lot.")
                                                # Passer le WCS de référence global, PAS le WCS généré individuel
                                                batch_data_for_drizzle_std.append( (img_d, hdr_d, self.reference_wcs_object) )
                                        
                                        if batch_data_for_drizzle_std:
                                            if self.drizzle_output_wcs is None: # Créer la grille de sortie Drizzle si pas encore fait
                                                ref_shape_for_driz_grid = self.memmap_shape[:2] if self.memmap_shape else reference_image_data_for_global_alignment.shape[:2]
                                                if self.reference_wcs_object is None: # Double check, devrait être attrapé avant
                                                    raise RuntimeError("WCS de référence global manquant pour créer la grille Drizzle.")
                                                self.drizzle_output_wcs, self.drizzle_output_shape_hw = self._create_drizzle_output_wcs(
                                                    self.reference_wcs_object, ref_shape_for_driz_grid, self.drizzle_scale
                                                )
                                            
                                            if self.drizzle_output_wcs: # Si la grille de sortie est prête
                                                self.stacked_batches_count += 1
                                                print(f"  DEBUG QM [_worker / Stacking Standard]: Appel _process_and_save_drizzle_batch pour lot Drizzle standard #{self.stacked_batches_count}")
                                                sci_path_batch, wht_paths_batch_list = self._process_and_save_drizzle_batch(
                                                    batch_data_for_drizzle_std, 
                                                    self.drizzle_output_wcs, 
                                                    self.drizzle_output_shape_hw, 
                                                    self.stacked_batches_count
                                                )
                                                if sci_path_batch and wht_paths_batch_list: 
                                                    self.intermediate_drizzle_batch_files.append((sci_path_batch, wht_paths_batch_list))
                                                    print(f"  DEBUG QM [_worker / Stacking Standard]: Lot Drizzle standard #{self.stacked_batches_count} sauvegardé.")
                                                else: 
                                                    self.failed_stack_count += len(batch_data_for_drizzle_std) # Compter les images du lot comme échec
                                                    print(f"  WARN QM [_worker / Stacking Standard]: Échec _process_and_save_drizzle_batch pour lot Drizzle standard #{self.stacked_batches_count}")
                                            else: # Erreur création grille sortie Drizzle
                                                raise RuntimeError("Échec création grille de sortie pour Drizzle standard par lot.")
                                        else: # Aucune donnée valide dans le lot pour Drizzle
                                            print(f"  WARN QM [_worker / Stacking Standard]: Aucune donnée valide dans le lot pour Drizzle standard.")

                                    else: # Stacking Classique (SUM/W) par lot
                                        self.stacked_batches_count += 1
                                        print(f"  DEBUG QM [_worker / Stacking Standard]: Appel _process_completed_batch pour lot classique #{self.stacked_batches_count}")
                                        self._process_completed_batch(
                                            current_batch_items_with_masks_for_stack_batch, 
                                            self.stacked_batches_count, 
                                            self.total_batches_estimated
                                        )
                                    # Vider le lot après traitement
                                    current_batch_items_with_masks_for_stack_batch = []
                            else: # Échec _process_file (alignement, etc.)
                                self.failed_align_count +=1 
                                print(f"  WARN QM [_worker / Stacking Standard]: Échec _process_file pour '{file_name_for_log}'. Ignoré.")
                                # Fichier déplacé vers unaligned par _process_file si échec

                    except Exception as e_process_file_specific: # Erreur dans le traitement du fichier spécifique
                        self.update_progress(f"⚠️ Erreur traitement fichier '{file_name_for_log}': {type(e_process_file_specific).__name__} - {e_process_file_specific}")
                        print(f"ERREUR QM [_worker V5.3 - TupleFix] (Traitement Fichier Spécifique '{file_name_for_log}'): {type(e_process_file_specific).__name__} - {e_process_file_specific}")
                        traceback.print_exc(limit=1)
                        self.failed_stack_count +=1 # Ou un compteur d'erreur général pour ce type d'erreur
                    
                    # Toujours appeler task_done pour l'item de la queue, même si erreur interne
                    self.queue.task_done() 
                
                except Empty: # La queue est vide
                    self.update_progress("ⓘ File d'attente vide. Vérification dernier lot partiel et dossiers supplémentaires...")
                    print("DEBUG QM [_worker V5.3 - TupleFix / Queue Vide]: File vide. Traitement dernier lot partiel / scan dossiers.")
                    
                    # --- Traiter dernier lot partiel (SI PAS Mosaïque Astrometry par panneau, car elle accumule différemment) ---
                    # Note: La branche Mosaïque Locale n'utilise PAS current_batch_items_with_masks_for_stack_batch.
                    #       Elle ajoute directement à all_aligned_files_with_info_for_mosaic.
                    if not use_astrometry_per_panel_mosaic and current_batch_items_with_masks_for_stack_batch:
                        print(f"  DEBUG QM [_worker / Queue Vide]: Traitement dernier lot partiel ({len(current_batch_items_with_masks_for_stack_batch)} items).")
                        if self.drizzle_active_session: # Drizzle Standard dernier lot
                            # ... (logique identique à la gestion du lot plein pour Drizzle standard) ...
                            batch_data_for_drizzle_std_last = []
                            for item_driz_std_l in current_batch_items_with_masks_for_stack_batch:
                                img_dl, hdr_dl, _scores_dl, _wcs_gen_dl, _mask_dl = item_driz_std_l
                                if img_dl is not None:
                                    if self.reference_wcs_object is None: raise RuntimeError("WCS réf global manquant pour Drizzle standard (dernier lot).")
                                    batch_data_for_drizzle_std_last.append( (img_dl, hdr_dl, self.reference_wcs_object) )
                            if batch_data_for_drizzle_std_last:
                                if self.drizzle_output_wcs is None: 
                                    ref_shape_driz_l = self.memmap_shape[:2] if self.memmap_shape else reference_image_data_for_global_alignment.shape[:2]
                                    if self.reference_wcs_object is None: raise RuntimeError("WCS réf global manquant pour grille Drizzle (dernier lot).")
                                    self.drizzle_output_wcs,self.drizzle_output_shape_hw=self._create_drizzle_output_wcs(self.reference_wcs_object,ref_shape_driz_l,self.drizzle_scale)
                                if self.drizzle_output_wcs: 
                                    self.stacked_batches_count+=1
                                    print(f"  DEBUG QM [_worker / Queue Vide]: Appel _process_and_save_drizzle_batch pour DERNIER lot Drizzle standard #{self.stacked_batches_count}")
                                    s_p_l,w_ps_l=self._process_and_save_drizzle_batch(batch_data_for_drizzle_std_last,self.drizzle_output_wcs,self.drizzle_output_shape_hw,self.stacked_batches_count)
                                    if s_p_l and w_ps_l: self.intermediate_drizzle_batch_files.append((s_p_l,w_ps_l)); print(f"  DEBUG QM [_worker / Queue Vide]: DERNIER lot Drizzle standard #{self.stacked_batches_count} sauvegardé.")
                                    else: self.failed_stack_count += len(batch_data_for_drizzle_std_last); print(f"  WARN QM [_worker / Queue Vide]: Échec _process_and_save_drizzle_batch DERNIER lot Drizzle std.")
                            else: print(f"  WARN QM [_worker / Queue Vide]: Aucune donnée valide pour DERNIER lot Drizzle std.")
                        
                        elif not self.drizzle_active_session: # Stacking Classique (SUM/W) dernier lot
                            self.stacked_batches_count+=1
                            print(f"  DEBUG QM [_worker / Queue Vide]: Appel _process_completed_batch pour DERNIER lot classique #{self.stacked_batches_count}")
                            self._process_completed_batch(current_batch_items_with_masks_for_stack_batch,self.stacked_batches_count,self.total_batches_estimated)
                        
                        current_batch_items_with_masks_for_stack_batch = [] # Vider le lot
                    
                    # --- Vérifier s'il y a des dossiers additionnels à traiter ---
                    folder_to_process_next = None 
                    with self.folders_lock:
                        if self.additional_folders: 
                            folder_to_process_next = self.additional_folders.pop(0)
                            self.update_progress(f"folder_count_update:{len(self.additional_folders)}") # Informer l'UI du changement de compteur
                    
                    if folder_to_process_next: 
                        self.current_folder = folder_to_process_next
                        self.update_progress(f"📂 Passage au dossier supplémentaire: {os.path.basename(folder_to_process_next)}")
                        print(f"DEBUG QM [_worker V5.3 - TupleFix / Queue Vide]: Passage au dossier suivant '{os.path.basename(folder_to_process_next)}'. Ajout de ses fichiers à la queue...")
                        self._add_files_to_queue(folder_to_process_next) # Ajouter les fichiers du nouveau dossier
                        self._recalculate_total_batches() # Recalculer l'estimation des lots
                        # La boucle while principale va continuer car folder_to_process_next était non-None
                    else: # Plus de fichiers dans la queue ET plus de dossiers additionnels
                        self.update_progress("✅ Fin de la file d'attente et des dossiers. Passage à la finalisation...")
                        print("DEBUG QM [_worker V5.3 - TupleFix / Queue Vide]: Plus de fichiers, plus de dossiers. Sortie de la boucle principale.")
                        break # Sortir de la BOUCLE 2 (while principale) pour passer à la finalisation
                
                except Exception as e_inner_loop_main: # Erreur dans le TRY principal de la boucle (ex: self.queue.get échoue autrement que par Empty)
                    self.update_progress(f"⚠️ Erreur majeure (non Empty) dans la boucle worker: {type(e_inner_loop_main).__name__} - {e_inner_loop_main}")
                    print(f"ERREUR QM [_worker V5.3 - TupleFix] (Boucle Principale Interne): {type(e_inner_loop_main).__name__} - {e_inner_loop_main}")
                    traceback.print_exc(limit=1)
                    # Si un fichier était en cours de traitement et qu'une erreur s'est produite ici,
                    # il faut s'assurer que task_done est appelé pour cet item si possible.
                    if file_path and self.queue.unfinished_tasks > 0:
                         try: 
                             self.queue.task_done()
                             print(f"  DEBUG QM [_worker / Erreur Boucle Principale]: task_done() appelé pour item potentiellement problématique.")
                         except ValueError: pass # Peut être déjà fait ou l'item n'était pas "get"
                
                finally: # Nettoyage pour CETTE itération de la BOUCLE 2
                    # print(f"DEBUG QM [_worker V5.3 - TupleFix / Fin Itération Boucle]: Nettoyage variables d'itération.")
                    # Supprimer les références aux variables de cette itération pour aider le GC
                    del file_path 
                    del aligned_data_item, header_item, quality_scores_item, wcs_object_indiv_item, valid_pixel_mask_item
                    # Suppression des variables spécifiques à la mosaïque locale
                    # if 'current_panel_data_loaded_from_fits' in locals(): del current_panel_data_loaded_from_fits
                    # if 'panel_header_for_processing' in locals(): del panel_header_for_processing
                    if 'current_panel_data_processed_for_align' in locals(): del current_panel_data_processed_for_align
                    if '_aligned_img_temp_from_local_aligner' in locals(): del _aligned_img_temp_from_local_aligner
                    if 'M_transform_from_local_aligner' in locals(): del M_transform_from_local_aligner
                    if 'align_success_local_aligner' in locals(): del align_success_local_aligner
                    if 'wcs_approx_panel_for_fallback' in locals(): del wcs_approx_panel_for_fallback
                    if 'M_matrix_from_wcs_fallback' in locals(): del M_matrix_from_wcs_fallback
                    
                    # Forcer un garbage collect occasionnellement
                    if self.processed_files_count > 0 and self.processed_files_count % 20 == 0: 
                        # print(f"DEBUG QM [_worker V5.3 - TupleFix / Fin Itération Boucle]: Appel gc.collect() (processed_files_count={self.processed_files_count}).")
                        gc.collect() 
            
            # === FIN DE LA BOUCLE 2 (while not self.stop_processing) ===
            # (La boucle a été quittée soit par un 'break' explicite, soit parce que self.stop_processing est devenu True)

            # === SECTION 3: TRAITEMENT FINAL APRÈS LA BOUCLE ===
            print(f"DEBUG QM [_worker V5.3 - TupleFix]: Sortie de la boucle principale. Début de la phase de finalisation...")
            # Log des états finaux importants avant la logique de finalisation
            print(f"  ÉTAT FINAL AVANT BLOC if/elif/else de finalisation:")
            print(f"    - self.stop_processing (flag d'arrêt utilisateur): {self.stop_processing}")
            print(f"    - self.is_mosaic_run: {self.is_mosaic_run}")
            if self.is_mosaic_run:
                print(f"      - Mode alignement mosaïque utilisé: '{self.mosaic_alignment_mode}'")
                print(f"      - Nombre d'items collectés pour mosaïque: {len(all_aligned_files_with_info_for_mosaic)}")
            print(f"    - self.drizzle_active_session (pour Drizzle standard): {self.drizzle_active_session}")
            if self.drizzle_active_session:
                print(f"      - Mode Drizzle standard: '{self.drizzle_mode}'")
                print(f"      - Nombre de fichiers intermédiaires Drizzle standard: {len(self.intermediate_drizzle_batch_files)}")
            print(f"    - self.images_in_cumulative_stack (pour stacking classique): {self.images_in_cumulative_stack}")
            
            # --- BRANCHE 3.A: Traitement interrompu par l'utilisateur ---
            if self.stop_processing: 
                self.update_progress("🛑 Traitement interrompu par l'utilisateur avant la sauvegarde finale complète.")
                print("DEBUG QM [_worker V5.3 - TupleFix / Finalisation]: Traitement interrompu par l'utilisateur.")
                # Tenter de sauvegarder un état partiel si pertinent
                # Pour le stacking classique (SUM/W), _save_final_stack gère stopped_early=True
                if not self.is_mosaic_run and not self.drizzle_active_session and self.images_in_cumulative_stack > 0:
                    self.update_progress("   -> Tentative de sauvegarde du stack classique partiel (SUM/W)...")
                    print("  DEBUG QM [_worker V5.3 - TupleFix / Interrompu]: Appel _save_final_stack pour SUM/W partiel.")
                    self._save_final_stack(output_filename_suffix="_sumw_stopped_partial", stopped_early=True)
                # Pour Drizzle standard, on ne combine pas les lots intermédiaires si arrêté
                elif self.drizzle_active_session and not self.is_mosaic_run and self.intermediate_drizzle_batch_files:
                     self.update_progress("   -> Drizzle standard interrompu. Pas de combinaison des lots intermédiaires.")
                     print("  DEBUG QM [_worker V5.3 - TupleFix / Interrompu]: Drizzle standard interrompu, pas de combinaison de lots.")
                     self.final_stacked_path = None # Assurer qu'aucun chemin final n'est défini
                # Pour la mosaïque, on ne fait pas d'assemblage final si arrêté
                elif self.is_mosaic_run:
                     self.update_progress("   -> Assemblage mosaïque interrompu.")
                     print("  DEBUG QM [_worker V5.3 - TupleFix / Interrompu]: Assemblage mosaïque interrompu.")
                     self.final_stacked_path = None
                else: # Aucun travail partiel significatif à sauvegarder
                     self.final_stacked_path = None
            
            # --- BRANCHE 3.B: Finalisation Mosaïque ---
            elif self.is_mosaic_run:
                print(f"DEBUG QM [_worker V5.3 - TupleFix / Finalisation]: Entrée dans la branche de finalisation Mosaïque.")
                print(f"  -> Mode d'alignement utilisé pour cette mosaïque: '{self.mosaic_alignment_mode}'")
                if not all_aligned_files_with_info_for_mosaic: 
                    self.processing_error = "Aucun panneau (ou image) aligné n'a été collecté pour la mosaïque."
                    self.update_progress(f"❌ ERREUR Mosaïque: {self.processing_error}")
                    print(f"  ERREUR QM [_worker / Mosaïque]: {self.processing_error}")
                else:
                    self.update_progress("🏁 Finalisation du traitement de la Mosaïque...")
                    # Réinitialiser la grille de sortie globale (sera recalculée spécifiquement pour la mosaïque)
                    self.drizzle_output_wcs = None
                    self.drizzle_output_shape_hw = None 
                    
                    # --- Calcul de la grille de sortie pour la mosaïque ---
                    # Si alignement local, la grille est calculée avec _calculate_local_mosaic_output_grid
                    # Si alignement Astrometry par panneau, elle est calculée avec _calculate_final_mosaic_grid_optimized
                    if use_local_aligner_for_this_mosaic_run: 
                        print("  DEBUG QM [_worker / Mosaïque]: Calcul de la grille de sortie pour mosaïque locale (OMBB)...")
                        if self.reference_wcs_object: # WCS d'ancrage doit être défini
                            self.drizzle_output_wcs, self.drizzle_output_shape_hw = self._calculate_local_mosaic_output_grid(
                                all_aligned_files_with_info_for_mosaic, # Contient (img_orig, hdr, wcs_ancre, M_trans, masque)
                                self.reference_wcs_object # Le WCS de l'ancre
                            )
                        else: # Ne devrait pas arriver si le panneau de réf a été traité
                            self.processing_error = "WCS d'ancrage local manquant pour calculer la grille de sortie mosaïque."
                            print(f"  ERREUR QM [_worker / Mosaïque]: {self.processing_error}")
                    
                    elif use_astrometry_per_panel_mosaic: 
                        print("  DEBUG QM [_worker / Mosaïque]: Calcul de la grille de sortie pour mosaïque Astrometry par panneau...")
                        # Extraire les WCS individuels et les shapes des panneaux
                        # all_aligned_files_with_info_for_mosaic contient (img_data, header, wcs_indiv_panel, valid_mask)
                        wcs_list_for_grid_calc = []
                        shapes_list_for_grid_calc = []
                        for item_astro_mosaic in all_aligned_files_with_info_for_mosaic:
                            if len(item_astro_mosaic) == 4 and item_astro_mosaic[0] is not None and item_astro_mosaic[2] is not None:
                                wcs_list_for_grid_calc.append(item_astro_mosaic[2]) # Le WCS individuel
                                shapes_list_for_grid_calc.append(item_astro_mosaic[0].shape[:2]) # Shape H,W de l'image
                        
                        if wcs_list_for_grid_calc and len(wcs_list_for_grid_calc) == len(shapes_list_for_grid_calc):
                            self.drizzle_output_wcs, self.drizzle_output_shape_hw = self._calculate_final_mosaic_grid_optimized(
                                wcs_list_for_grid_calc, 
                                shapes_list_for_grid_calc, 
                                self.drizzle_scale # Utiliser l'échelle Drizzle globale pour la mosaïque
                            )
                        else: 
                            self.processing_error = "Données WCS ou shapes manquantes/incohérentes pour calculer la grille de sortie mosaïque Astrometry."
                            print(f"  ERREUR QM [_worker / Mosaïque]: {self.processing_error}")
                    
                    # Vérifier si la grille a été calculée
                    if self.drizzle_output_wcs is None or self.drizzle_output_shape_hw is None:
                        self.processing_error = self.processing_error or "Échec critique du calcul de la grille de sortie pour la mosaïque."
                        self.update_progress(f"❌ ERREUR Mosaïque: {self.processing_error}")
                        print(f"  ERREUR QM [_worker / Mosaïque]: {self.processing_error}")
                    else: # Grille OK, lancer l'assemblage de la mosaïque
                        print(f"  DEBUG QM [_worker / Mosaïque]: Grille de sortie mosaïque prête. Shape HW: {self.drizzle_output_shape_hw}. Appel process_mosaic_from_aligned_files...")
                        pm_func = None # Import tardif
                        try: 
                            from ..enhancement.mosaic_processor import process_mosaic_from_aligned_files as pmf
                            pm_func = pmf
                            print("DEBUG QM [_worker V5.3 - TupleFix]: Import dynamique process_mosaic_from_aligned_files RÉUSSI.")
                        except ImportError as e_pmf:
                             print(f"ERREUR QM [_worker V5.3 - TupleFix]: Import dynamique process_mosaic_from_aligned_files ÉCHOUÉ: {e_pmf}")
                        
                        if pm_func:
                            # process_mosaic_from_aligned_files s'attend à (list_info, instance_queued_stacker, callback)
                            # Il utilisera self.drizzle_output_wcs et self.drizzle_output_shape_hw qui sont maintenant définis
                            final_mosaic_data_normalized_01, final_mosaic_header = pm_func(
                                all_aligned_files_with_info_for_mosaic, 
                                self, # Passer l'instance actuelle de SeestarQueuedStacker
                                self.update_progress
                            )
                            if final_mosaic_data_normalized_01 is not None and final_mosaic_header is not None:
                                # Sauvegarder la mosaïque finale
                                mosaic_filename = os.path.join(self.output_folder, "stack_final_mosaic_drizzle.fit") 
                                self.update_progress(f"   -> Sauvegarde de la mosaïque finale : {os.path.basename(mosaic_filename)}...")
                                print(f"  DEBUG QM [_worker / Mosaïque]: Sauvegarde FITS mosaïque: {mosaic_filename}")
                                save_fits_image(final_mosaic_data_normalized_01, mosaic_filename, final_mosaic_header, overwrite=True)
                                self.final_stacked_path = mosaic_filename
                                # Stocker pour l'aperçu final dans _processing_finished
                                self.last_saved_data_for_preview = final_mosaic_data_normalized_01.copy() 
                                self.update_progress("   -> Mosaïque finale sauvegardée avec succès.")
                            else: 
                                self.processing_error = self.processing_error or "Échec de l'assemblage final de la mosaïque (process_mosaic_from_aligned_files a retourné None)."
                                self.update_progress(f"❌ ERREUR Mosaïque: {self.processing_error}")
                                print(f"  ERREUR QM [_worker / Mosaïque]: {self.processing_error}")
                        else: 
                            self.processing_error = "Module d'assemblage de mosaïque (process_mosaic_from_aligned_files) non importable."
                            self.update_progress(f"❌ ERREUR CRITIQUE Mosaïque: {self.processing_error}")
                            print(f"  ERREUR QM [_worker / Mosaïque]: {self.processing_error}")
            
            # --- BRANCHE 3.C: Finalisation Drizzle Standard (non-mosaïque) ---
            elif self.drizzle_active_session: # Et pas self.is_mosaic_run (implicite par elif)
                print(f"DEBUG QM [_worker V5.3 - TupleFix / Finalisation]: Entrée dans la branche de finalisation Drizzle Standard (Mode: {self.drizzle_mode}).")
                self.update_progress(f"🏁 Finalisation Drizzle Standard (Mode {self.drizzle_mode})...")
                
                if self.drizzle_mode == "Final":
                    if self.intermediate_drizzle_batch_files:
                        print(f"  DEBUG QM [_worker / Drizzle Standard Final]: Combinaison de {len(self.intermediate_drizzle_batch_files)} lots Drizzle intermédiaires.")
                        # _combine_intermediate_drizzle_batches prend (list_fichiers_interm, wcs_sortie, shape_sortie_hw)
                        # Le WCS et Shape de sortie devraient déjà être définis (self.drizzle_output_wcs / _shape_hw)
                        if self.drizzle_output_wcs is None or self.drizzle_output_shape_hw is None:
                             raise RuntimeError("Grille de sortie Drizzle Standard (self.drizzle_output_wcs/_shape_hw) non définie pour la combinaison finale.")
                        
                        final_sci_drizzle_combined_hxwxc, final_wht_drizzle_combined_hxwxc = self._combine_intermediate_drizzle_batches(
                            self.intermediate_drizzle_batch_files, 
                            self.drizzle_output_wcs, 
                            self.drizzle_output_shape_hw
                        )
                        if final_sci_drizzle_combined_hxwxc is not None and final_wht_drizzle_combined_hxwxc is not None:
                            print(f"  DEBUG QM [_worker / Drizzle Standard Final]: Combinaison des lots Drizzle réussie.")
                            # Préparer le header pour _save_final_stack
                            self.current_stack_header = self._update_header_for_drizzle_final() 
                            
                            print(f"  DEBUG QM [_worker / Drizzle Standard Final]: Appel _save_final_stack pour Drizzle Final.")
                            self._save_final_stack(
                                output_filename_suffix="_drizzle_final", 
                                stopped_early=False, 
                                drizzle_final_sci_data=final_sci_drizzle_combined_hxwxc, 
                                drizzle_final_wht_data=final_wht_drizzle_combined_hxwxc
                            )
                        else: 
                            self.processing_error = f"Échec combinaison finale des lots Drizzle (Mode Final)."
                            self.update_progress(f"❌ ERREUR Drizzle Standard: {self.processing_error}"); 
                            print(f"  ERREUR QM [_worker / Drizzle Standard Final]: {self.processing_error}")
                            self.final_stacked_path = None
                    else: 
                        self.update_progress(f"   -> Aucun lot Drizzle intermédiaire à combiner pour Drizzle (Mode Final). Stack final non créé."); 
                        print(f"  WARN QM [_worker / Drizzle Standard Final]: Aucun lot Drizzle intermédiaire trouvé.")
                        self.final_stacked_path = None
                
                elif self.drizzle_mode == "Incremental":
                    # Pour Drizzle Incrémental, le résultat est déjà dans les accumulateurs SUM/W
                    # _save_final_stack lira depuis ces accumulateurs.
                    # On doit juste s'assurer que current_stack_header est à jour.
                    print(f"  DEBUG QM [_worker / Drizzle Standard Incrémental]: Préparation pour sauvegarde depuis accumulateurs SUM/W.")
                    if self.images_in_cumulative_stack > 0 or (self.cumulative_sum_memmap is not None and np.any(self.cumulative_sum_memmap)):
                        # Le header self.current_stack_header a été mis à jour par _process_incremental_drizzle_batch
                        # ou initialisé correctement.
                        if self.current_stack_header is None: # Sécurité, ne devrait pas arriver
                             self.current_stack_header = self._update_header_for_drizzle_final() # Fallback
                             self.current_stack_header['STACKTYP'] = (f'Drizzle Incr SUM/W ({self.drizzle_scale:.0f}x)', 'Stacking method')
                        
                        print(f"  DEBUG QM [_worker / Drizzle Standard Incrémental]: Appel _save_final_stack pour Drizzle Incrémental (SUM/W).")
                        self._save_final_stack(output_filename_suffix="_drizzle_incr_sumw", stopped_early=False)
                        # _save_final_stack utilisera drizzle_final_sci/wht_data=None, donc lira depuis memmap.
                    else:
                        self.update_progress("   -> Aucun Drizzle Incrémental (SUM/W) à sauvegarder (0 images/poids accumulés)."); 
                        print(f"  WARN QM [_worker / Drizzle Standard Incrémental]: Pas de données Drizzle Incrémental à sauvegarder.")
                        self.final_stacked_path = None
            
            # --- BRANCHE 3.D: Finalisation Stacking Classique (non-mosaïque, non-Drizzle) ---
            elif not self.is_mosaic_run and not self.drizzle_active_session: 
                print(f"DEBUG QM [_worker V5.3 - TupleFix / Finalisation]: Entrée dans la branche de finalisation Stacking Classique (SUM/W).")
                self.update_progress("🏁 Finalisation du stacking classique (SUM/W)...")
                # Le résultat est dans les accumulateurs SUM/W. _save_final_stack les lira.
                if self.images_in_cumulative_stack > 0 or (self.cumulative_sum_memmap is not None and np.any(self.cumulative_sum_memmap)):
                    # self.current_stack_header a été mis à jour par _process_completed_batch.
                    print(f"  DEBUG QM [_worker / Stacking Classique]: Appel _save_final_stack pour SUM/W classique. Images accumulées: {self.images_in_cumulative_stack}")
                    self._save_final_stack(output_filename_suffix="_classic_sumw", stopped_early=False)
                else: 
                    self.update_progress("   -> Aucune image accumulée pour le stacking classique. Stack final non créé."); 
                    print(f"  WARN QM [_worker / Stacking Classique]: Pas de données SUM/W à sauvegarder.")
                    self.final_stacked_path = None
            
            else:  # Cas imprévu (ne devrait pas arriver si la logique des flags est correcte)
                self.processing_error = "État de finalisation non géré (combinaison de flags inattendue)."
                self.update_progress(f"❌ ERREUR INTERNE: {self.processing_error}")
                print(f"  ERREUR QM [_worker / Finalisation]: {self.processing_error}")
                
        except RuntimeError as rte: # Erreurs d'exécution critiques levées explicitement
             self.processing_error = str(rte)
             print(f"ERREUR QM [_worker V5.3 - TupleFix] (Runtime Exception Critique): {rte}")
             traceback.print_exc(limit=1)
             # L'update_progress est fait par l'endroit qui lève le RuntimeError
        except Exception as e_global_worker: # Erreurs globales inattendues dans le worker
            self.processing_error = f"Erreur inattendue dans le thread worker: {type(e_global_worker).__name__} - {e_global_worker}"
            print(f"ERREUR QM [_worker V5.3 - TupleFix] (Exception Globale Inattendue): {type(e_global_worker).__name__} - {e_global_worker}")
            traceback.print_exc(limit=2) # Afficher plus de détails pour les erreurs inattendues
            self.update_progress(f"❌ ERREUR CRITIQUE Worker: {self.processing_error}")
        
        finally: # FINALLY du TRY principal du worker (celui qui englobe la boucle while et la finalisation)
            print("DEBUG QM [_worker V5.3 - TupleFix]: Entrée dans le bloc FINALLY principal du worker.") 
            
            # Fermer les memmaps s'ils ont été ouverts (fait par _close_memmaps)
            # _save_final_stack (pour SUM/W) appelle déjà _close_memmaps.
            # Si Drizzle/Mosaïque, les memmaps n'étaient peut-être pas utilisés pour l'accumulation finale,
            # mais ils ont été créés. Assurons-nous qu'ils sont fermés.
            # (Si _save_final_stack n'a pas été appelé, ou si c'était pour Drizzle avec données directes)
            if self.cumulative_sum_memmap is not None or self.cumulative_wht_memmap is not None:
                 print("  DEBUG QM [_worker / Finally]: Appel _close_memmaps (par sécurité)...")
                 self._close_memmaps() 
            
            # Nettoyage des fichiers temporaires
            if self.perform_cleanup: 
                self.update_progress("🧹 Nettoyage final des fichiers temporaires...")
                print("  DEBUG QM [_worker / Finally]: Exécution nettoyage des fichiers temporaires...")
                self.cleanup_unaligned_files()      # Fichiers qui ont échoué à l'alignement
                self.cleanup_temp_reference()       # ref_image.fit/png dans temp_processing
                self._cleanup_drizzle_temp_files()  # Dossier drizzle_temp_inputs (contient aligned_input_xxxx.fits)
                self._cleanup_drizzle_batch_outputs() # Dossier drizzle_batch_outputs (contient batch_xxx_sci/wht.fits)
                self._cleanup_mosaic_panel_stacks_temp() # Dossier mosaic_panel_stacks_temp (ancienne logique ou si utilisé)
                
                # Nettoyage des fichiers memmap PHYSIQUES si SUM/W classique (déjà fait dans _save_final_stack si cleanup activé)
                # Mais si on a une erreur AVANT _save_final_stack, ils peuvent rester.
                # _close_memmaps ne supprime que les références, pas les fichiers.
                memmap_dir_final = os.path.join(self.output_folder, "memmap_accumulators") if self.output_folder else None
                if self.sum_memmap_path and os.path.exists(self.sum_memmap_path): 
                    try: os.remove(self.sum_memmap_path); print("     -> Fichier SUM.npy (worker finally) supprimé.")
                    except Exception as e_rm_sum: print(f"     WARN: Erreur suppression SUM.npy (worker finally): {e_rm_sum}")
                if self.wht_memmap_path and os.path.exists(self.wht_memmap_path):
                    try: os.remove(self.wht_memmap_path); print("     -> Fichier WHT.npy (worker finally) supprimé.") 
                    except Exception as e_rm_wht: print(f"     WARN: Erreur suppression WHT.npy (worker finally): {e_rm_wht}")
                try: # Essayer de supprimer le dossier memmap s'il est vide
                    if memmap_dir_final and os.path.isdir(memmap_dir_final) and not os.listdir(memmap_dir_final): 
                        os.rmdir(memmap_dir_final)
                        print(f"     -> Dossier memmap vide (worker finally) supprimé: {memmap_dir_final}")
                except Exception as e_rmdir_mem: print(f"     WARN: Erreur suppression dossier memmap (worker finally): {e_rmdir_mem}")
            else: # perform_cleanup est False
                 self.update_progress("ⓘ Nettoyage final ignoré. Fichiers temporaires et memmap conservés.")
                 print("  DEBUG QM [_worker / Finally]: perform_cleanup est False, pas de nettoyage.")

            # Vider les listes pour libérer mémoire (prudent de vérifier leur existence au cas où)
            if 'current_batch_items_with_masks_for_stack_batch' in locals() and current_batch_items_with_masks_for_stack_batch is not None: 
                del current_batch_items_with_masks_for_stack_batch
            if 'all_aligned_files_with_info_for_mosaic' in locals() and all_aligned_files_with_info_for_mosaic is not None: 
                del all_aligned_files_with_info_for_mosaic
            if hasattr(self, 'intermediate_drizzle_batch_files') and self.intermediate_drizzle_batch_files is not None: 
                self.intermediate_drizzle_batch_files = [] # Vider la liste
            
            # Assurer que le flag de processing_active est bien à False à la fin
            self.processing_active = False 
            # Stocker un flag pour que l'UI sache si l'arrêt était dû à self.stop_processing (demande utilisateur)
            # ou à une fin normale/erreur.
            self.stop_processing_flag_for_gui = self.stop_processing 
            
            print("  DEBUG QM [_worker / Finally]: Vidage listes et GC...")
            gc.collect()
            print("DEBUG QM [_worker V5.3 - TupleFix]: Fin du bloc FINALLY principal. Flag processing_active mis à False.")
            self.update_progress("🚪 Thread de traitement principal terminé.")
            # Le thread va se terminer naturellement ici.
            # L'UI (via _track_processing_progress) détectera que le worker n'est plus is_running()
            # et appellera _processing_finished() dans le thread GUI.

    # ... (reste des méthodes de la classe : _calculate_M_from_wcs, _calculate_local_mosaic_output_grid, etc.)
