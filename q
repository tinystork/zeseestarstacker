[33mca954fe[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mWIP[m[33m, [m[1;31morigin/WIP[m[33m)[m Fix histogram scaling and add log toggle
[33m937d2f2[m Merge pull request #498 from tinystork/codex/déplacer-calcul-histogramme-dans-un-thread
[33maa1473b[m Async histogram computation
[33m65e758e[m Merge pull request #497 from tinystork/codex/dé-surcharger-callback-gui-dans-queue_manager.py
[33m5ec9070[m Throttle GUI progress updates
[33m582619b[m requirements
[33m4788171[m requirements
[33m42d75ef[m update requirement.txt
[33mef80e30[m Merge pull request #496 from tinystork/codex/corriger-l-utilisation-du-gpu-pour-phase-5
[33mbb1132f[m Fix GPU selection logic for phase 5
[33m4b1ad2e[m Merge pull request #495 from tinystork/codex/fix-process_error-on-cpu-only-run
[33m263f5ee[m fix gpu init
[33mddf5d02[m Merge pull request #494 from tinystork/codex/activer-utilisation-gpu-phase-5
[33m0675dff[m Enable GPU stacking option
[33mbf44939[m Merge pull request #493 from tinystork/codex/corriger-l-importation-de-cupy-et-fixations-cuda
[33m1c3efb4[m Merge branch 'WIP' into codex/corriger-l-importation-de-cupy-et-fixations-cuda
[33me8ad3ed[m revert seestar mods and delay gpu init
[33mc8433ec[m refactor: delay cupy import until after device selection
[33mba8b453[m Merge pull request #491 from tinystork/codex/améliorer-la-détection-gpu-dans-zemosaic
[33m4166bc0[m Enhance GPU detection and selection
[33me50bd67[m ajout wmi
[33mb601ba2[m Merge pull request #490 from tinystork/codex/ajouter-un-sélecteur-de-gpu-avec-combobox
[33m9204c81[m Add GPU selector combobox and CuPy device support
[33ma08c411[m Merge pull request #489 from tinystork/codex/ajouter-wrapper-gpu/cpu-pour-stacking
[33m0e62399[m Add GPU wrappers for stacking
[33mf695e06[m Merge pull request #488 from tinystork/codex/instrumenter-eta-et-callbacks-de-progression
[33mfc25f03[m Add stage progress callbacks and ETA tracking
[33m09b1d9e[m Merge pull request #487 from tinystork/codex/ajouter-prise-en-charge-gpu/cuda-avec-fallback-cpu
[33mdb1d517[m Merge branch 'WIP' into codex/ajouter-prise-en-charge-gpu/cuda-avec-fallback-cpu
[33m9c1f36b[m Add GPU wrapper and integration
[33m8f2918b[m Add GPU-enabled reprojection with CPU fallback
[33m26e0905[m Merge pull request #485 from tinystork/codex/ajouter-option-gpu-nvidia-phase-5
[33m9f91409[m Add GPU selection option for phase 5
[33m6d89c09[m Merge pull request #483 from tinystork/codex/ajouter-prise-en-charge-gpu-nvidia-phase-5
[33m5761a7c[m Add GPU option for final mosaic assembly
[33m3c95273[m remove unused files
[33m77d81ec[m remove unused files
[33md47c187[m Merge pull request #481 from tinystork/codex/analyser-la-lenteur-du-traitement-winsorized-sigma-clip
[33m16a5971[m Optimize winsorized sigma clipping
[33mccf094d[m Merge pull request #480 from tinystork/codex/résoudre-problème-de-stacking-avant-reproject
[33m0dcca00[m Remove auto_limit_memory_fraction
[33m5518d2f[m Merge pull request #479 from tinystork/codex/nettoyer-zemosaic_worker.py
[33mf29578a[m fix worker wcs pretreat
[33mcd02ef3[m Merge pull request #478 from tinystork/codex/réparer-fonction-clustering-dans-zemosaic_worker.py
[33m0896542[m fix: reinject WCS and restore clustering
[33m01bdb5b[m check
[33mf78571f[m Merge pull request #477 from tinystork/codex/réparer-la-fonction-clustering-dans-zemosaic_worker.py
[33m4f1cee2[m Fix Seestar stack clustering
[33m7129fec[m reference
[33md20d714[m Merge pull request #476 from tinystork/codex/réparer-fonction-clustering-dans-zemosaic_worker.py
[33m42b7e6d[m Restore robust Seestar clustering
[33m1f0eaf7[m Merge pull request #475 from tinystork/codex/réparer-fonction-clustering-dans-zemosaic_worker.py
[33m5897f5e[m Revert Seestar clustering to stable logic
[33m771aaa0[m worker_old
[33m4b24983[m Merge pull request #474 from tinystork/codex/mettre-à-jour-seuil-de-clustering-et-gestion-crval
[33mec4734b[m Merge branch 'WIP' into codex/mettre-à-jour-seuil-de-clustering-et-gestion-crval
[33m1ea790b[m Fix logging aliases and clustering fallback
[33m09dbc2d[m Merge pull request #473 from tinystork/codex/mettre-à-jour-seuil-de-clustering-et-gestion-crval
[33m334d081[m Fix Seestar stack clustering fallback
[33mc3b4079[m Merge pull request #472 from tinystork/codex/corriger-erreur-module-zemosaic_worker
[33m09d5e5d[m Implement run_hierarchical_mosaic_process wrapper
[33m0ddd166[m Merge pull request #471 from tinystork/codex/corriger-le-clustering-après-l-appel-à-astap
[33mf55dd27[m fix clustering center fallback
[33m0255a68[m update
[33md687c72[m Merge pull request #469 from tinystork/codex/vérifier-et-rétablir-appel-clustering-master-tiles
[33mea8ce2c[m Use GUI clustering threshold
[33mad222c2[m threshold
[33md76a9ef[m threshold
[33m84a54c4[m Merge pull request #468 from tinystork/codex/mettre-à-jour-compteur-progressif-des-master-tiles
[33m965d07c[m Fix master tile progress updates
[33m12a62f8[m Merge pull request #467 from tinystork/codex/implement-robust-wcs-solving-for-astap
[33mf3e2037[m Add luminance option and normalize solver input
[33m07eb2b1[m Merge pull request #466 from tinystork/duh2nt-codex/propager-les-champs-ra-et-dec-dans-le-fichier-fits
[33m9dcaadb[m Merge branch 'WIP' into duh2nt-codex/propager-les-champs-ra-et-dec-dans-le-fichier-fits
[33m8ef01fa[m Propagate RA/DEC fallback and test
[33m804a7a0[m Merge pull request #465 from tinystork/codex/propager-les-champs-ra-et-dec-dans-le-fichier-fits
[33m77aa40a[m Propagate RA/DEC to final stack
[33m7fa9f6a[m Merge pull request #464 from tinystork/codex/corriger-l-erreur-de-convolution-dans-le-traitement-d-image
[33mad56073[m Handle 3-channel FITS images for astrometry.net
[33m7d4af5b[m Merge pull request #463 from tinystork/codex/corriger-transmission-de-solver_settings-au-worker
[33m1b73504[m Fix worker import to use package path
[33m8ce00af[m Merge pull request #462 from tinystork/codex/corriger-transmission-de-solver_settings-au-worker
[33m7e02e7d[m Fix solver settings propagation and add test
[33m5978ba5[m Fix passing solver settings
[33m3e101a7[m Merge pull request #459 from tinystork/codex/mettre-en-place-interrogation-astrometry-dans-zemosaic_gui
[33m02a43b1[m Improve astrometry solver logging and API key
[33m5e371a6[m Merge pull request #458 from tinystork/codex/corriger-la-gestion-des-solveurs-dans-l-interface-et-le-work
[33m2c3730a[m Pass solver settings to worker process
[33mcc75cbe[m Merge pull request #457 from tinystork/codex/forcer-la-résolution-par-astrometry
[33m4121ac5[m Merge branch 'WIP' into codex/forcer-la-résolution-par-astrometry
[33m096553b[m Improve astrometry diagnostics
[33mabd53b5[m Merge pull request #456 from tinystork/codex/forcer-la-résolution-par-astrometry
[33m1b71dc3[m Improve astrometry failure logging
[33m8159c54[m Merge pull request #455 from tinystork/codex/restaurer-hiérarchie-solveurs-locaux
[33m7533703[m Add ansvr solver option
[33m0059ecf[m Merge pull request #454 from tinystork/codex/intégrer-astrometry_solver-dans-zemosaic
[33m335c921[m Add astroquery-based web solver
[33mb50209b[m Merge pull request #453 from tinystork/codex/refactoriser-getwcs_pretreat-pour-gérer-astrometry-et-astap
[33m021687f[m Add test ensuring astrometry solver used
[33m1cb366b[m Merge pull request #452 from tinystork/codex/refactoriser-getwcs_pretreat-pour-gérer-astrometry-et-astap
[33m363a4c2[m refactor solver selection
[33mab79482[m Merge pull request #451 from tinystork/codex/intégrer-astrometry.net-dans-zemosaic
[33mfb8a183[m Add Astrometry.net solver integration
[33m7304302[m Merge pull request #450 from tinystork/codex/corriger-l-appel-à-astrometry-dans-le-gui
[33m58eedea[m Fix astrometry solver usage
[33m003363e[m Merge pull request #449 from tinystork/codex/corriger-l-erreur-de-gestionnaire-de-géométrie
[33maca9ec5[m fix solver frame toggling
[33m0d2768f[m Merge pull request #448 from tinystork/codex/mettre-à-jour-la-gestion-des-solveurs
[33m9fa7d67[m Add Astrometry solver fallback and GUI frame toggle
[33m3b4f84e[m Merge pull request #447 from tinystork/codex/corriger-l-erreur-de-chargement-de-zemosaicgui
[33ma079eab[m Fix GUI import in run_zemosaic
[33m7c7120b[m Merge pull request #446 from tinystork/codex/ajouter-les-paramètres-astrometry.net-dans-l-ui
[33mbe73f91[m Add solver settings dataclass and GUI controls
[33mcc5de2c[m Merge pull request #445 from tinystork/codex/filtrer-les-avertissements-de-noyau-dans-drizzle_integration
[33me0f4f6e[m Ignore flux conserving kernel warnings
[33mc337d68[m Merge pull request #444 from tinystork/codex/filtrer-avertissements-de-noyau-non-conservateur-de-flux
[33m25318e7[m Filter flux-conserving kernel warnings
[33m7c2e5c0[m Merge pull request #443 from tinystork/codex/implement-cumulative-drizzle-preview
[33mfbd429d[m feat: add cumulative preview for incremental drizzle
[33m6634421[m Merge pull request #442 from tinystork/codex/ajouter-une-auto-correction-dynamique-pour-drizzle
[33mb3812da[m Add dynamic drizzle preview auto-stretch
[33mad59816[m Merge pull request #441 from tinystork/codex/implémenter-un-aperçu-normalisé-en-temps-réel
[33me715dd2[m Add normalized preview method
[33m48d8646[m Merge pull request #440 from tinystork/codex/vérifier-et-corriger-le-rendu-de-drizzle
[33mcc355df[m Fix incremental drizzle preview
[33m1af7d4a[m Merge pull request #439 from tinystork/codex/paralléliser-l-utilisation-de-drizzle-sur-gpu-et-processeur
[33m160a1ee[m Add optional GPU support for drizzle utils
[33m70a4496[m Merge pull request #438 from tinystork/codex/ajouter-option-gpu-dans-gui-pour-drizzle
[33mcffc0a6[m Add GPU toggle for Drizzle
[33m1943a3e[m Merge pull request #437 from tinystork/codex/corriger-le-gel-du-gui-pendant-un-traitement-drizzle
[33meba632a[m Fix GUI freeze by removing print debug statements
[33mf369647[m Merge pull request #436 from tinystork/codex/améliorer-réactivité-mode-drizzle
[33mf3fa700[m Run Drizzle processing in separate thread
[33m0746e15[m Merge pull request #435 from tinystork/codex/corriger-affichage-image-dans-gui
[33m784a7d1[m Fix preview handling and update mosaic worker
[33mf27c152[m Merge pull request #434 from tinystork/codex/corriger-image-finale-sombre-et-supprimer-option-redondante
[33m7a68e74[m Remove drizzle flux renormalization option
[33m0f9fd7a[m Merge pull request #433 from tinystork/codex/corriger-perte-de-flux-stacking-drizzle
[33m2ea9b55[m Fix drizzle double normalization
[33m6ccc840[m Merge pull request #432 from tinystork/codex/corriger-l-image-illisible-en-mode-drizzle
[33mabf17ff[m Merge branch 'graphic-Beta' into codex/corriger-l-image-illisible-en-mode-drizzle
[33mcef11d0[m Handle 2D drizzle weights with CHW science
[33mb35ce8c[m Merge pull request #431 from tinystork/codex/corriger-l-image-illisible-en-mode-drizzle
[33mcad90f2[m Handle both weight map orientations in drizzle final
[33m86120b3[m Revert "Fix drizzle final weight broadcasting"
[33m33e7bf3[m Merge pull request #430 from tinystork/codex/corriger-l-image-illisible-en-mode-drizzle
[33m85d6fa7[m Fix drizzle final weight broadcasting
[33mb1b29aa[m Merge pull request #429 from tinystork/codex/corriger-l-erreur-axiserror-numpy
[33m2c90de3[m Fix axis error in Drizzle final mode
[33m3dd8abd[m Merge pull request #428 from tinystork/codex/accumulate-weight-map-in-seestarqueuedstacker
[33m28fb148[m Add cumulative weight map aggregation
[33md8b98c3[m Merge pull request #427 from tinystork/codex/supprimer-dossier-classic_batch_outputs-après-traitement
[33m31cc0e1[m cleanup classic batch outputs
[33m99437d4[m version 5.0.0 zemosaic 2.0.0 included
[33mf9c5c28[m Merge WIP into graphic-Beta
[33mbbb1eb0[m version 5.0.0 zemosaic version 2.0.0 included
[33md84dcf0[m Merge pull request #426 from tinystork/codex/ajouter-un-bouton--apply-snr-rejection
[33mf25ac5f[m Add Apply SNR button in visualization window
[33mb214b97[m Merge pull request #425 from tinystork/codex/ajouter-un-rangeslider-pour-snr
[33m404f7c4[m Add SNR RangeSlider and pending action handler
[33m9a450a6[m Merge pull request #424 from tinystork/codex/intégrer-le-traitement-parallèle-dans-le-pipeline-d-analyse
[33mf9f787f[m Integrate parallel processing
[33m9e74aed[m Merge pull request #423 from tinystork/codex/ajouter-logs-et-fiabiliser-winsorized-sigma-clip
[33m60e5137[m Log winsorized sigma clip result
[33m28b9e35[m Merge pull request #422 from tinystork/codex/ajouter-logs-et-fiabiliser-winsorized-sigma-clip
[33m5d8197b[m Add winsorized sigma clip logging and fix argument passing
[33m18b0fb7[m Merge pull request #421 from tinystork/codex/déplacer--enable-inter-batch-reprojection--dans-final-combin
[33m17489d0[m Move inter-batch reprojection option
[33mbd80d0a[m Merge pull request #420 from tinystork/codex/ajouter-option-winsorized-sigma-clip
[33m21186af[m Add winsorized sigma clip final combine option
[33mca76501[m Merge pull request #419 from tinystork/codex/ajouter-limite-ram-hq-dans-l-ui
[33m554e44d[m Add HQ RAM limit setting and combine logic
[33mf9357e9[m Merge pull request #418 from tinystork/codex/ajouter-option-zoom-roi-numérique
[33m8e54a6d[m Add zoom ROI option
[33m9b53886[m[33m ([m[1;31morigin/unstable[m[33m)[m Version 4.7.0 support dual gpus
[33mbae6b21[m Version 4.7.0 support dual gpus
[33mf9d2e17[m Merge pull request #417 from tinystork/codex/corriger-erreur-d-importation-de-modules
[33me132bec[m Fix relative imports for GUI modules
[33m9593d5c[m Merge pull request #416 from tinystork/codex/corriger-l-erreur-d-importation-relative
[33mf9e71a3[m Fix ZeMosaic GUI import when launched directly
[33m0bd1b67[m Fix direct execution for run_zemosaic
[33m568d327[m Merge pull request #415 from tinystork/codex/corriger-l-erreur-d-importation-relative
[33m31bcd6b[m Fix direct execution for run_zemosaic
[33m1a7fdfd[m Merge pull request #414 from tinystork/codex/corriger-l-erreur-d-importation-relative
[33m28d5f94[m Fix main script import when run as standalone
[33m4857635[m Merge pull request #413 from tinystork/codex/-title----corriger-l-erreur-modulenotfound
[33m1a18184[m Fix relative imports in CLI scripts
[33m22db821[m Merge pull request #412 from tinystork/bsyd98-codex/créer-et-intégrer-cuda_utils.py-pour-gestion-gpu-nvidia
[33m30f93f9[m Merge branch 'graphic-Beta' into bsyd98-codex/créer-et-intégrer-cuda_utils.py-pour-gestion-gpu-nvidia
[33mc1bc47a[m Fix GPU util import
[33m66791d3[m Merge pull request #411 from tinystork/codex/créer-et-intégrer-cuda_utils.py-pour-gestion-gpu-nvidia
[33m8ab499c[m Merge branch 'graphic-Beta' into codex/créer-et-intégrer-cuda_utils.py-pour-gestion-gpu-nvidia
[33m471b72c[m Fix GPU import path
[33m6c80476[m Merge pull request #410 from tinystork/codex/créer-et-intégrer-cuda_utils.py-pour-gestion-gpu-nvidia
[33m7ae91a1[m Add GPU enforcement utility and integrate in launch scripts
[33m827ecec[m version 4.5.0 fix standard stacking and zemosaic solver calls
[33md3b3e64[m Merge pull request #409 from tinystork/codex/corriger-comportement-de-zemosaic-avec-astap
[33m7c60126[m[33m ([m[1;31morigin/codex/corriger-comportement-de-zemosaic-avec-astap[m[33m)[m Stop astrometry fallback and rename unaligned folder
[33m577b9ff[m Merge pull request #408 from tinystork/codex/vérifier-options-de-post-traitement
[33ma41296e[m Apply feathering and low WHT mask
[33m27f8765[m Merge pull request #407 from tinystork/codex/corriger-appel-de-reproject-dans-le-log
[33m85f275a[m Respect user setting for inter-batch reprojection
[33m6e19225[m Merge pull request #406 from tinystork/codex/ajouter-de-nouvelles-fonctionnalités-de-stacking-et-reprojec
[33m006d45c[m Add continuous accumulation stacking
[33m198cc68[m[33m ([m[1;32munstable[m[33m)[m Merge pull request #405 from tinystork/codex/corriger-mise-à-jour-du-preview-en-stacking-classique
[33m4c54998[m Fix preview updates when reprojection between batches
[33m9f9e7da[m Merge pull request #404 from tinystork/codex/modifier-comportement-de-référence-wcs
[33mb429acc[m Respect freeze_reference_wcs when reprojecting
[33me60a751[m Merge pull request #403 from tinystork/codex/proposer-correction-pour-mode-stacking-classique
[33mf50189e[m Merge branch 'graphic-Beta' into codex/proposer-correction-pour-mode-stacking-classique
[33maf741ae[m Fix inter-batch reference solving
[33md748507[m Merge pull request #402 from tinystork/codex/proposer-correction-pour-mode-stacking-classique
[33m9f8f12d[m fix: respect freeze_reference_wcs when reprojection enabled
[33m211b260[m Merge pull request #401 from tinystork/codex/vérifier-propagation-mode-de-staking
[33m769622a[m Propagate stacking method
[33m7db3aa1[m Merge pull request #400 from tinystork/codex/mettre-à-jour-readme.md-avec-nouvelles-options
[33mbdd33cb[m docs: describe constant WCS inter-batch reprojection
[33m915f51c[m Merge pull request #399 from tinystork/codex/introduce-helper-for-fixed-orientation-grid
[33m2c6fd3b[m Add fixed orientation grid helper and tests
[33mfbed724[m Merge pull request #398 from tinystork/codex/add-solve_batches-flag-to-seestarqueuedstacker
[33m72468e2[m Add solve_batches flag and adjust batch solving
[33m8884192[m Merge pull request #397 from tinystork/codex/aligner-toutes-les-images-sur-la-référence
[33meb5ed4c[m Add option to freeze reference WCS
[33macba26f[m Merge pull request #396 from tinystork/codex/réajuster-taille-image-reprojetée
[33m60b82f1[m Fix batch reprojection size
[33m9d25010[m Merge pull request #395 from tinystork/codex/corriger-l-erreur-valueerror-fits-header
[33mad45b3e[m Fix NaN weight max causing FITS header error
[33md2d40e7[m Merge pull request #394 from tinystork/codex/corriger-erreur-de-validation-hdu-astropy
[33m2fae5e7[m Ignore FITS header verification
[33m3875496[m Merge pull request #393 from tinystork/codex/corriger-problème-d-alignement-et-reproject-entre-lots
[33mb5a0497[m fix interbatch reprojection wcs
[33md61b1ca[m Merge pull request #392 from tinystork/codex/corriger-erreur-runtimeerror-dans-le-thread
[33m47b5221[m fix autotuner thread restart
[33mb1a133a[m Merge pull request #391 from tinystork/codex/modifier-comportement-stacking-classique-pour-astap
[33maeb8d80[m Fix batch mask selection for inter-batch reprojection
[33m97aabec[m Merge pull request #390 from tinystork/rcusfs-codex/modifier-comportement-reproject_between_batches
[33m5be39b2[m Fix inter-batch reprojection batch handling
[33m06165d4[m Merge pull request #389 from tinystork/codex/modifier-comportement-reproject_between_batches
[33m5180b78[m Refactor batch reprojection workflow
[33mbfbee88[m Merge branch 'unstable' into graphic-Beta
[33m055772a[m Revert "Allow restarting autotuner thread"
[33m2d90722[m Merge pull request #388 from tinystork/codex/modifier-le-pré-scan-wcs-pour-le-mode-mosaïque
[33m5bb799c[m[33m ([m[1;31morigin/codex/modifier-le-pré-scan-wcs-pour-le-mode-mosaïque[m[33m)[m Limit WCS prescan to mosaic mode
[33m0de9ec1[m Merge pull request #387 from tinystork/codex/réparer-mode-stacking-classique-reproject_between_batches
[33ma9cd01e[m[33m ([m[1;31morigin/codex/réparer-mode-stacking-classique-reproject_between_batches[m[33m)[m Improve inter-batch reprojection logic
[33m59a3a0c[m Merge pull request #386 from tinystork/codex/rétablir-workflow-mode-stacking-classique
[33mf9a9ea7[m[33m ([m[1;31morigin/codex/rétablir-workflow-mode-stacking-classique[m[33m)[m Fix interbatch solver in classic reprojection mode
[33mb24ccef[m Merge pull request #385 from tinystork/aigmtg-codex/reprojeter-les-lots-empilés-avec-astap
[33m6fbe229[m[33m ([m[1;31morigin/aigmtg-codex/reprojeter-les-lots-empilés-avec-astap[m[33m)[m Update classic stacking reference workflow
[33m652e5d3[m Merge pull request #384 from tinystork/codex/reprojeter-les-lots-empilés-avec-astap
[33m67c09a4[m[33m ([m[1;31morigin/codex/reprojeter-les-lots-empilés-avec-astap[m[33m)[m Update classic stacking reference workflow
[33mca8997e[m Merge pull request #383 from tinystork/codex/refactor-mode-empilement-incrémental
[33mdbb298b[m[33m ([m[1;31morigin/codex/refactor-mode-empilement-incrémental[m[33m)[m Refactor inter-batch reprojection
[33m5c4aebe[m Merge branch 'graphic-Beta' into unstable
[33mb7332f3[m fix thread déja démarré
[33m32094dd[m Merge pull request #382 from tinystork/codex/corriger-erreur-thread-déjà-démarré
[33m59e01aa[m[33m ([m[1;31morigin/codex/corriger-erreur-thread-déjà-démarré[m[33m)[m Allow restarting autotuner thread
[33m645b0a8[m Merge branch 'graphic-Beta' into unstable
[33ma60e499[m version 4.2.9 allow resume stack
[33m9eb2743[m Merge pull request #381 from tinystork/codex/ajouter-auto-tune-cpu/i-o
[33m00b0bb8[m[33m ([m[1;31morigin/codex/ajouter-auto-tune-cpu/i-o[m[33m)[m Add CPU/IO autotuner
[33m0071e20[m Merge pull request #380 from tinystork/codex/implement-horizontal-qsplitter-with-state-saving
[33m1510a88[m[33m ([m[1;31morigin/codex/implement-horizontal-qsplitter-with-state-saving[m[33m)[m Use splitter for viewer layout
[33me85d2af[m Merge pull request #379 from tinystork/nxqku8-codex/optimiser-réactivité-du-gui
[33m1b4a38d[m[33m ([m[1;31morigin/nxqku8-codex/optimiser-réactivité-du-gui[m[33m)[m Merge branch 'graphic-Beta' into nxqku8-codex/optimiser-réactivité-du-gui
[33md4a8117[m Fix import issues with multiprocessing
[33mb526e7a[m Merge pull request #378 from tinystork/codex/optimiser-réactivité-du-gui
[33m200a1a6[m[33m ([m[1;31morigin/codex/optimiser-réactivité-du-gui[m[33m)[m Use process pool for winsorized stacking
[33md6c3631[m Merge pull request #377 from tinystork/codex/sécuriser-les-mises-à-jour-tkinter-depuis-des-threads
[33m457ed3a[m[33m ([m[1;31morigin/codex/sécuriser-les-mises-à-jour-tkinter-depuis-des-threads[m[33m)[m fix(gui): route progress and preview updates to main thread
[33m31c7090[m Merge pull request #376 from tinystork/codex/supprimer-les-anciens-fichiers-stack_batch
[33m36e4356[m[33m ([m[1;31morigin/codex/supprimer-les-anciens-fichiers-stack_batch[m[33m)[m Delete old stack partial after new one
[33m131ea79[m Merge pull request #375 from tinystork/codex/vérifier-et-améliorer-l-application-des-modes-d-empilement-e
[33md35e64c[m[33m ([m[1;31morigin/codex/vérifier-et-améliorer-l-application-des-modes-d-empilement-e[m[33m)[m Integrate quality weighting via dropdown
[33m68e4f53[m Merge pull request #374 from tinystork/codex/vérifier-et-améliorer-les-logs-d-assemblage
[33m1fe013c[m[33m ([m[1;31morigin/codex/vérifier-et-améliorer-les-logs-d-assemblage[m[33m)[m Add stacking normalization functions and modes
[33m7b95cca[m Merge pull request #373 from tinystork/codex/optimiser-_stack_batch-avec-multithread
[33m4847e5e[m[33m ([m[1;31morigin/codex/optimiser-_stack_batch-avec-multithread[m[33m)[m Optimize _stack_batch with multithreaded numpy
[33m12e3c0f[m Merge pull request #372 from tinystork/codex/exclure-effacement-fichiers-temporaire-et-créer-fichier-de-c
[33m622021b[m[33m ([m[1;31morigin/codex/exclure-effacement-fichiers-temporaire-et-créer-fichier-de-c[m[33m)[m feat: preserve reference and export run config
[33m1d17f46[m Merge pull request #371 from tinystork/doha3z-codex/ajouter-champ--last-stack-treated--et-reprise
[33m4d68b65[m[33m ([m[1;31morigin/doha3z-codex/ajouter-champ--last-stack-treated--et-reprise[m[33m)[m Merge branch 'graphic-Beta' into doha3z-codex/ajouter-champ--last-stack-treated--et-reprise
[33md63673d[m Save partial stacks after each batch
[33m98311ab[m Add resume stacking support and GUI field
[33md6fad8e[m Merge pull request #370 from tinystork/codex/ajouter-champ--last-stack-treated--et-reprise
[33mcf1d7c9[m[33m ([m[1;31morigin/codex/ajouter-champ--last-stack-treated--et-reprise[m[33m)[m Add resume stacking support and GUI field
[33m70ed544[m Version 4.2.5  Move alrerady stacked files to a separate folder
[33mff709cc[m Merge pull request #369 from tinystork/codex/corriger-l-erreur-d-argument-dupliqué-dans-queue_manager.py
[33m737a879[m[33m ([m[1;31morigin/codex/corriger-l-erreur-d-argument-dupliqué-dans-queue_manager.py[m[33m)[m Fix duplicate params in start_processing
[33m08bb78c[m Merge pull request #368 from tinystork/4llb29-codex/ajouter-déplacement-de-fichiers-après-chaque-lot
[33mc41d088[m[33m ([m[1;31morigin/4llb29-codex/ajouter-déplacement-de-fichiers-après-chaque-lot[m[33m)[m Merge branch 'graphic-Beta' into 4llb29-codex/ajouter-déplacement-de-fichiers-après-chaque-lot
[33m488f2c2[m Fix duplicate parameter syntax in start_processing
[33mc85ef80[m Merge pull request #367 from tinystork/codex/ajouter-déplacement-de-fichiers-après-chaque-lot
[33mfa6f79c[m Merge branch 'graphic-Beta' into codex/ajouter-déplacement-de-fichiers-après-chaque-lot
[33m05cbd8b[m Fix duplicate move_stacked arg
[33mc0e220f[m Merge pull request #366 from tinystork/xoicq5-codex/ajouter-déplacement-de-fichiers-après-chaque-lot
[33m661ae2c[m Merge branch 'graphic-Beta' into xoicq5-codex/ajouter-déplacement-de-fichiers-après-chaque-lot
[33m3b47a3e[m feat: move stacked files and partial save
[33m01d2c7c[m Merge pull request #364 from tinystork/codex/ajouter-déplacement-de-fichiers-après-chaque-lot
[33m0358f88[m feat: move raws after batch and save partial stacks
[33m4d4788f[m Version 4.2.0 drizzle fixed
[33m4ce5533[m Merge branch 'graphic-Beta'
[33m812ef62[m Fix FITS unsigned 16 handling
[33m7eae5a6[m Merge pull request #360 from tinystork/codex/identifier-problème-d-images-sombres-avec-drizzle
[33m310addc[m self.drizzle_wht_threshold = 0.0
[33me1f6182[m Fix renormalize_fits to update data
[33mb0f86b6[m Merge pull request #359 from tinystork/codex/analyser-problème-drizzle-final
[33ma08cb11[m Fix drizzle final pixmap bounds
[33m3830815[m Merge pull request #358 from tinystork/codex/ajouter-renormalisation-flux-dans-_merge_drizzle_batches
[33m86087e1[m Add drizzle stack renormalization
[33me1bae43[m Merge pull request #357 from tinystork/codex/fix-warning-kernel-lanczos2-not-flux-conserving
[33mca9f948[m fix FITS unsigned header
[33me9fce0b[m Merge pull request #356 from tinystork/codex/fix-drizzle-stacking-flux-normalization
[33mab8543b[m feat: add drizzle renormalization option
[33m79c48a8[m Merge branch 'graphic-Beta' V 4.1.0
[33m3101f00[m Merge pull request #355 from tinystork/codex/paralléliser-la-reproduction-avec-gpu
[33m98190a1[m Add parallel reprojection pipeline
[33m2fd797e[m version 4.1.0 improved reproject canvas
[33m3a5b441[m Merge pull request #354 from tinystork/codex/corriger-erreur-oserror-dans-queue_manager.py
[33m1e9b5ad[m check reference shape dimensions
[33mc755cbf[m Merge pull request #353 from tinystork/2cmtdk-codex/implémenter-calcul-robuste-de-grille-dans-queue_manager.py
[33m4245f78[m Merge branch 'graphic-Beta' into 2cmtdk-codex/implémenter-calcul-robuste-de-grille-dans-queue_manager.py
[33m518ffa8[m Fix indentation in prescan loop
[33m822e313[m Merge pull request #352 from tinystork/cnqrbc-codex/implémenter-calcul-robuste-de-grille-dans-queue_manager.py
[33md28021f[m Merge branch 'graphic-Beta' into cnqrbc-codex/implémenter-calcul-robuste-de-grille-dans-queue_manager.py
[33mce9e7a0[m Run backend start in worker thread
[33m8e83ac9[m Merge pull request #351 from tinystork/ru3ltf-codex/implémenter-calcul-robuste-de-grille-dans-queue_manager.py
[33m09d147b[m Merge branch 'graphic-Beta' into ru3ltf-codex/implémenter-calcul-robuste-de-grille-dans-queue_manager.py
[33md627180[m Fix prescan order
[33me9e6d8a[m Merge pull request #350 from tinystork/fku8yf-codex/implémenter-calcul-robuste-de-grille-dans-queue_manager.py
[33m97e3295[m Handle reprojection pre-scan
[33m439098e[m Merge pull request #349 from tinystork/codex/implémenter-calcul-robuste-de-grille-dans-queue_manager.py
[33m456a4c6[m Improve grid calc and cropping
[33m71cc8fc[m Merge pull request #348 from tinystork/codex/garantir-cohérence-des-memmaps-avec-reference_shape
[33mce01a6d[m Fix memmap shape after global grid
[33mbec8a2d[m Merge pull request #347 from tinystork/boxwyo-codex/fix-problème-d’image-noire-à-cause-du-premier-lot
[33m2172191[m Merge branch 'graphic-Beta' into boxwyo-codex/fix-problème-d’image-noire-à-cause-du-premier-lot
[33mbde5d8a[m Ensure consistent target shape
[33m88dc42b[m Merge pull request #346 from tinystork/codex/fix-problème-d’image-noire-à-cause-du-premier-lot
[33m6df5b47[m Fix first batch reprojection and enforce pixel shape
[33mdcfe91a[m Merge pull request #343 from tinystork/codex/corriger-l-erreur-de-valueerror-dans-reproject_and_coadd
[33m30467b0[m Handle reprojection failures
[33m87a73af[m Merge branch 'graphic-Beta'
[33m522701c[m version 4.0.0
[33m8d8428f[m Merge pull request #342 from tinystork/codex/rétablir-alignement-inter-master_tiles
[33mb3af1eb[m zemosaic: fix master tile alignment
[33m619ca79[m Merge pull request #341 from tinystork/codex/remplacer-reproject_interp-par-reproject_and_coadd
[33m6b7772a[m Implement batch reproject using reproject_and_coadd
[33m25046b1[m[33m ([m[1;31morigin/version-3.9.0[m[33m, [m[1;32mversion-3.9.0[m[33m)[m version 3.9.0
[33m96cb196[m Merge pull request #340 from tinystork/codex/corriger-l-erreur-valueerror-dans-reproject_and_combine
[33m386ea80[m fix shape mismatch in incremental reprojection
[33me7f1af3[m Merge pull request #338 from tinystork/codex/vérifier-calcul-de-la-grille-de-projection
[33m08c288f[m queue manager: use optimal wcs calc
[33m93924d4[m Use local mosaic grid for inter-batch drizzle
[33m4c8b0df[m Merge pull request #336 from tinystork/codex/mettre-à-jour-_calculate_final_mosaic_grid
[33m2202133[m Add scale factor to mosaic grid calc
[33m4420880[m Merge pull request #335 from tinystork/codex/aligner-comportement-des-solveurs-astrométriques
[33maa9e517[m Allow Astrometry.net solving with API key
[33m3623afc[m Merge pull request #334 from tinystork/codex/update-solver-related-settings-for-third-party-disable
[33m8b9d144[m Disable solver usage when third-party disabled
[33ma23159d[m Merge pull request #333 from tinystork/v3ehpg-codex/ajouter-réglage--use-third-party-solver
[33m548a4b7[m Merge branch 'graphic-Beta' into v3ehpg-codex/ajouter-réglage--use-third-party-solver
[33m0799723[m Keep solver disable toggle state
[33m126c700[m Merge pull request #332 from tinystork/codex/ajouter-réglage--use-third-party-solver
[33me2f2aae[m feat: allow disabling third-party solvers
[33m6b9b487[m Merge pull request #331 from tinystork/codex/extend-start_processing-to-support-winsor_limits
[33m55fb587[m Add winsorized sigma clipping support
[33m18da5c5[m Merge pull request #330 from tinystork/codex/ajout-de-l-option-apply_rewinsor
[33m3f049c8[m Remove GUI toggle for re-winsorization
[33m5a75714[m Merge pull request #329 from tinystork/codex/mettre-à-jour-la-méthode-_process_file
[33m6b1232a[m Fix classic stacking regression by restoring size consistency
[33m7b57986[m Merge pull request #323 from tinystork/codex/adapter-toile-en-mode-stack-classique-+-reproject
[33m0abfafd[m feat: auto crop classic reproject
[33m6ca434f[m Merge pull request #322 from tinystork/ox3thx-codex/fix-cadre-et-reprojection-incrémentale
[33m6629db4[m Merge branch 'graphic-Beta' into ox3thx-codex/fix-cadre-et-reprojection-incrémentale
[33m8720d26[m Fix fixed grid usage
[33m6eb796e[m Merge pull request #321 from tinystork/b0oc9n-codex/fix-cadre-et-reprojection-incrémentale
[33m67663b2[m Fix missing reference shape when grid pre-scan fails
[33m0d5942d[m Merge pull request #320 from tinystork/codex/fix-cadre-et-reprojection-incrémentale
[33m2d22abc[m Add fixed grid computation for incremental reprojection
[33mdeeb0cf[m Merge pull request #319 from tinystork/wxue2n-codex/refactor-seestarqueuedstacker-for-parallel-processing
[33m5e6f1bb[m Merge branch 'graphic-Beta' into wxue2n-codex/refactor-seestarqueuedstacker-for-parallel-processing
[33m0c0f38c[m Fix thread config import and update tests
[33ma16a2d0[m Merge pull request #318 from tinystork/codex/refactor-seestarqueuedstacker-for-parallel-processing
[33m42d28ff[m Add thread configuration and batch helpers
[33m08b1fba[m Merge pull request #317 from tinystork/codex/corriger-troncature-image-dans-mode-stacking+reproject
[33ma7dcab1[m Fix cropping when reprojection with single WCS
[33m1433297[m Merge pull request #316 from tinystork/codex/refactor-fonction-warp_image
[33mfca5487[m Refactor warp_image with dynamic canvas
[33mca5cd4b[m Merge pull request #315 from tinystork/codex/mise-à-jour-de-la-gestion-des-memmaps
[33m171deb5[m Handle dynamic memmap resizing
[33m6609743[m Merge pull request #314 from tinystork/codex/implement-image-alignment-transformation
[33m678ecd0[m Add dynamic warp size computation
[33m2e426f1[m Merge pull request #313 from tinystork/codex/restructurer-la-logique-de-reprojection-et-des-lots
[33ma5f1cb2[m Refactor worker reprojection logic
[33m0604b1c[m Merge pull request #312 from tinystork/codex/paralléliser-la-fonction-resolve_all_wcs
[33m12ca576[m Add parallel WCS reprojection utility
[33m9e3e435[m Merge pull request #311 from tinystork/codex/corriger-calcul-crpix-pour-mosaïque-centrée-sur-ombb
[33m746c78d[m Fix preview WCS centering
[33m3488584[m Merge pull request #310 from tinystork/codex/ajouter-aperçu-dynamique-dans-queue_manager.py
[33m7847742[m Improve preview cropping
[33mb7b2531[m Merge pull request #309 from tinystork/codex/implémenter-pré-solving-global-pour-la-reprojection-des-imag
[33m0190add[m Implement global WCS solving
[33m208e166[m Merge pull request #308 from tinystork/codex/implémenter-pré-scan-global-et-calcul-de-grille-finale
[33md3743ca[m Fix global WCS grid calculation
[33mcafa9a1[m Merge pull request #307 from tinystork/fhghws-codex/remplacer-taille-globale-fixe-par-calcul-dynamique
[33m1ccfa6d[m Fix offset for dynamic mosaic grid
[33mdcf5a52[m Merge pull request #306 from tinystork/e0zwkk-codex/remplacer-taille-globale-fixe-par-calcul-dynamique
[33m173038a[m Use global mosaic shape for reprojection canvas
[33m62d1b09[m Merge pull request #305 from tinystork/codex/remplacer-taille-globale-fixe-par-calcul-dynamique
[33me8ff2ff[m Update progress message for dynamic grid
[33mde5e1a7[m Merge pull request #304 from tinystork/8i75qi-codex/implémenter-gestion-de-grille-wcs-globale
[33m148ae57[m Merge branch 'graphic-Beta' into 8i75qi-codex/implémenter-gestion-de-grille-wcs-globale
[33m700c652[m Use dynamic bounding box for global WCS
[33m991b415[m Merge pull request #303 from tinystork/oatp4g-codex/implémenter-gestion-de-grille-wcs-globale
[33m54584ba[m Merge branch 'graphic-Beta' into oatp4g-codex/implémenter-gestion-de-grille-wcs-globale
[33me9ec36a[m Fix global WCS grid init
[33m3c5a975[m Merge pull request #302 from tinystork/codex/implémenter-gestion-de-grille-wcs-globale
[33m25e6976[m feat: add global WCS grid prescan
[33me318855[m Merge pull request #301 from tinystork/aworsy-codex/appliquer-les-changements-pour-la-gestion-de-mosaïque-fixe
[33ma43e28c[m fix mosaic frame orientation
[33m807443b[m Merge pull request #300 from tinystork/codex/appliquer-les-changements-pour-la-gestion-de-mosaïque-fixe
[33mcceb42d[m Fix mosaic finalization and grid
[33md9ac20a[m Merge pull request #299 from tinystork/codex/corriger-bug-échelle-et-implémenter-cadrage-portrait
[33m3da4f2a[m Fix WCS derotation bug and improve mosaic grid
[33m17fa4f9[m Merge pull request #298 from tinystork/codex/mettre-à-jour-_finalize_mosaic_processing-avec-reproject_and
[33me3025bb[m Use reproject_and_coadd for mosaic finalization
[33mc127f94[m Merge pull request #297 from tinystork/codex/supprimer-les-print--ajout-critique
[33maa603d6[m Remove critical debug prints
[33m480e235[m Merge pull request #296 from tinystork/codex/modifier-queue_manager.py-et-supprimer-un-bloc
[33me476632[m Remove forced WCS derotation
[33mea80a7d[m Merge pull request #295 from tinystork/codex/supprimer-méthodes-obsolètes-dans-queue_manager.py
[33med69751[m Remove deprecated mosaic grid methods
[33mfc2bb4d[m Merge pull request #294 from tinystork/codex/unifier-logique-de-création-de-grille
[33md209583[m Fix reference WCS rotation handling
[33m9f28cd7[m Merge pull request #293 from tinystork/codex/remplacer-méthode-_calculate_final_mosaic_grid
[33ma047e2e[m Implement custom mosaic grid
[33m2e7ae05[m Merge pull request #292 from tinystork/codex/implémenter-ajustement-automatique-intelligent
[33m5203dad[m feat(gui): preserve manual stretch settings
[33m789e3de[m rotate in stack classic false
[33mc7ff461[m Merge pull request #291 from tinystork/codex/corriger-initialisation-de-current_stack_header
[33m6f41ec2[m Fix header init for reproject batches
[33m0f64ec2[m Merge pull request #290 from tinystork/codex/corriger-le-pipeline-de-reprojection
[33m6c1c805[m Fix reprojection batch handling
[33mc8eb2c5[m Merge pull request #289 from tinystork/codex/analyser-échec-wcs-lot-5
[33m93831c6[m Improve WCS solving for stacked batches
[33m9071f71[m Merge pull request #288 from tinystork/codex/corriger-wcs-pour-lot-empilé
[33me2eba58[m Fix WCS solving for stacked batches
[33md40f0df[m Merge pull request #287 from tinystork/codex/modifier-astap_downsample-dans-queue_manager.py
[33mc5b111c[m Update docstring for dynamic ASTAP downsample
[33m52e4d62[m Merge pull request #286 from tinystork/codex/add-subframes-for-astap-settings
[33m2cb502b[m Add ASTAP search options to local solver GUI
[33mc00b436[m Merge pull request #285 from tinystork/codex/résolution-astrométrique-échouée-pour-plusieurs-fichiers-fit
[33m80798cf[m Remove fallback alignment for reprojection mode
[33m6bc5787[m Revert "Merge pull request #284 from tinystork/codex/ajouter-méthode-_solve_stacked_batch"
[33m4962023[m Merge pull request #284 from tinystork/codex/ajouter-méthode-_solve_stacked_batch
[33m803ba56[m Ajoute résolution dédiée pour lots empilés
[33m6a0e950[m Merge pull request #283 from tinystork/codex/résolution-astrométrique-échouée-pour-certaines-images
[33m3c1b7fa[m Simplify batch stacking using numpy
[33m75cbfee[m Merge pull request #282 from tinystork/codex/fix-manque-de-wcs-dans-certains-lots
[33m0340e26[m Add astroalign fallback when astrometry fails
[33m9e824df[m Merge pull request #281 from tinystork/codex/traitement-et-résolution-wcs-des-images-fits
[33m47f85dc[m Fix inter-batch reprojection WCS handling
[33mbb2836d[m Merge pull request #280 from tinystork/codex/refactor-reproject_and_combine-function
[33m96272c2[m Refactor incremental reprojection stacking
[33m18a1430[m Merge pull request #279 from tinystork/codex/mettre-à-jour-le-mode-is_classic_reproject_mode
[33m8b26d15[m Add classic reproject mode and tests
[33m7150e00[m Merge pull request #278 from tinystork/codex/troubleshoot-wcs-reference-missing-error
[33md397e47[m Fix WCS initialization for batch reprojection
[33mf3b8325[m Merge pull request #277 from tinystork/codex/mettre-à-jour-les-appels-à-_process_completed_batch
[33m30983b2[m Add comment clarifying reference WCS parameter
[33md956287[m Merge pull request #276 from tinystork/codex/mettre-à-jour-les-appels-dans-_worker
[33mb037615[m Fix reference WCS argument passing
[33m1bb798a[m Merge pull request #275 from tinystork/ttmn8r-codex/corriger-logique-d-initialisation-dans-_process_completed_ba
[33m59cbef0[m Merge branch 'graphic-Beta' into ttmn8r-codex/corriger-logique-d-initialisation-dans-_process_completed_ba
[33m2767533[m Remove initialize_master export
[33m48a6683[m Merge pull request #274 from tinystork/94h1uq-codex/corriger-logique-d-initialisation-dans-_process_completed_ba
[33macb4720[m Merge branch 'graphic-Beta' into 94h1uq-codex/corriger-logique-d-initialisation-dans-_process_completed_ba
[33m6f33613[m Pass reference WCS explicitly
[33m3ebd604[m Merge pull request #273 from tinystork/codex/corriger-logique-d-initialisation-dans-_process_completed_ba
[33me393373[m Fix reprojection master stack initialization
[33me07a000[m Merge pull request #272 from tinystork/codex/corriger-la-résolution-wcs-dans-_process_completed_batch
[33m0826463[m Use first image WCS for batch reprojection
[33madaa8ec[m Merge pull request #271 from tinystork/codex/ajouter-méthode-_solve_stacked_batch-et-modifier-traitement
[33ma5f3d35[m feat(queue): add solver helper for stacked batches
[33m6316052[m Merge pull request #270 from tinystork/codex/modifier-traitement-de-lot-avec-solveur
[33m14da6d8[m Use solve_image_wcs for batch reprojection
[33m612acab[m Merge pull request #269 from tinystork/codex/update-_process_completed_batch-logic
[33m548b23e[m Handle batch solve failures during inter-batch reprojection
[33md63a81e[m Merge pull request #268 from tinystork/codex/update-queue_manager.py-for-reproject_between_batches
[33m343679f[m Fix incremental reprojection item handling
[33md99631b[m Merge pull request #267 from tinystork/codex/créer-module-d-aide-et-étendre-seestarqueuedstacker
[33m51fc78b[m Add incremental reprojection support
[33md8d5a6b[m[33m ([m[1;31morigin/version-3.3.0[m[33m, [m[1;32mversion-3.3.0[m[33m)[m version 3.3.0 beta reproject classic incremental stack
[33m87813c7[m Merge pull request #266 from tinystork/codex/refactor-reproject-between-batches-feature
[33md8829ea[m refactor reproject workflow
[33m4ddbda0[m Revert "Merge pull request #265 from tinystork/codex/refactoriser-le-mode-de-reprojection-entre-lots"
[33mab7895a[m Merge pull request #265 from tinystork/codex/refactoriser-le-mode-de-reprojection-entre-lots
[33m4891c76[m[33m ([m[1;31morigin/codex/refactoriser-le-mode-de-reprojection-entre-lots[m[33m)[m Refactor reproject workflow
[33m8751f4d[m Merge pull request #264 from tinystork/0g1d86-codex/résolution-astrométrique-et-traitement-drizzle
[33maed870c[m[33m ([m[1;31morigin/0g1d86-codex/résolution-astrométrique-et-traitement-drizzle[m[33m)[m Merge branch 'graphic-Beta' into 0g1d86-codex/résolution-astrométrique-et-traitement-drizzle
[33me755610[m Fallback to manual grid if reproject grid fails
[33m8fb71b3[m Merge pull request #263 from tinystork/w4lftk-codex/résolution-astrométrique-et-traitement-drizzle
[33m26e32f6[m[33m ([m[1;31morigin/w4lftk-codex/résolution-astrométrique-et-traitement-drizzle[m[33m)[m Merge branch 'graphic-Beta' into w4lftk-codex/résolution-astrométrique-et-traitement-drizzle
[33m018ad6a[m Copy NAXIS dims for failed batch solves
[33mef13564[m Merge pull request #262 from tinystork/69v0in-codex/résolution-astrométrique-et-traitement-drizzle
[33mc90361c[m Handle single WCS case in grid calc
[33m04fd586[m Merge pull request #261 from tinystork/codex/résolution-astrométrique-et-traitement-drizzle
[33m32caf25[m[33m ([m[1;31morigin/codex/résolution-astrométrique-et-traitement-drizzle[m[33m)[m Fix inter-batch reprojection
[33mb89689b[m Merge pull request #260 from tinystork/codex/répliquer-logique-de-zemosaic_worker-dans-queue_manager
[33m1872302[m[33m ([m[1;31morigin/codex/répliquer-logique-de-zemosaic_worker-dans-queue_manager[m[33m)[m queue manager: add header-based pixel_shape and reproject fallback
[33mc9b30d8[m Merge pull request #259 from tinystork/codex/refactor-queue_manager.py-to-remove-redundant-reprojection
[33mb24139b[m[33m ([m[1;31morigin/codex/refactor-queue_manager.py-to-remove-redundant-reprojection[m[33m)[m Avoid duplicate batch reprojection
[33m54640c2[m Merge pull request #258 from tinystork/codex/réviser-la-fonction-_calculate_final_mosaic_grid
[33mf86be05[m[33m ([m[1;31morigin/codex/réviser-la-fonction-_calculate_final_mosaic_grid[m[33m)[m fallback pixel_shape from header
[33m961d762[m Merge pull request #257 from tinystork/codex/update-wcs-parsing-and-add-regression-test
[33mfdf0ee9[m[33m ([m[1;31morigin/codex/update-wcs-parsing-and-add-regression-test[m[33m)[m Ensure WCS celestial parsing
[33m325f7fb[m Merge pull request #256 from tinystork/codex/corriger-l-erreur-de-typeerror-dans-le-stack
[33m95bb20d[m[33m ([m[1;31morigin/codex/corriger-l-erreur-de-typeerror-dans-le-stack[m[33m)[m Use local stacking to avoid ZeMosaic call
[33m24bb047[m Merge pull request #255 from tinystork/codex/update-_process_file-with-astrometry-solver
[33mf5635bc[m[33m ([m[1;31morigin/codex/update-_process_file-with-astrometry-solver[m[33m)[m Add astrometry branch for reproject and tests
[33m1730470[m Merge pull request #254 from tinystork/codex/ajouter-des-messages-de-mise-à-jour-avant-chaque-return
[33m7cb827a[m[33m ([m[1;31morigin/codex/ajouter-des-messages-de-mise-à-jour-avant-chaque-return[m[33m)[m log skipped reproject
[33m75bc816[m Merge pull request #253 from tinystork/codex/appliquer-np.nan_to_num-pour-nettoyer-les-nan
[33m43a2f21[m[33m ([m[1;31morigin/codex/appliquer-np.nan_to_num-pour-nettoyer-les-nan[m[33m)[m Handle NaNs in coverage maps
[33m7b68b89[m Merge pull request #252 from tinystork/codex/extend-reprojection-in-_save_and_solve_classic_batch
[33m6344091[m[33m ([m[1;31morigin/codex/extend-reprojection-in-_save_and_solve_classic_batch[m[33m)[m Reproject weight maps when saving classic batches
[33mbffaba6[m Merge pull request #251 from tinystork/codex/ajouter-reprojection-entre-lots-dans-le-mode-classique
[33m504c80d[m[33m ([m[1;31morigin/codex/ajouter-reprojection-entre-lots-dans-le-mode-classique[m[33m)[m Fix inter-batch reprojection logic
[33m9665db4[m Revert "Merge pull request #250 from tinystork/codex/fix-échec-de-résolution-astrométrique-en-mode-classique"
[33m950e7da[m Merge pull request #250 from tinystork/codex/fix-échec-de-résolution-astrométrique-en-mode-classique
[33m1cac131[m[33m ([m[1;31morigin/codex/fix-échec-de-résolution-astrométrique-en-mode-classique[m[33m)[m Use ZeMosaic ASTAP solver for classic batch
[33me8376d0[m Merge pull request #249 from tinystork/codex/add-astap_downsample-and-astap_sensitivity-settings
[33me0a2305[m[33m ([m[1;31morigin/codex/add-astap_downsample-and-astap_sensitivity-settings[m[33m)[m Add ASTAP downsample and sensitivity settings
[33md15dc4e[m Merge pull request #248 from tinystork/codex/ajouter-liaison-checkbox-reproject_between_batches
[33m988eb2f[m[33m ([m[1;31morigin/codex/ajouter-liaison-checkbox-reproject_between_batches[m[33m)[m Fix fallback WCS reprojection indentation
[33m46c5bf2[m Merge remote-tracking branch 'origin/graphic-Beta' into unstable
[33m32ad829[m Version 3.2.1 all set except standard stack with reproject
[33m659e15b[m Merge pull request #247 from tinystork/codex/add-bp/wp-line-persistence-in-histogram
[33m66f015f[m Preserve histogram range after final stack
[33ma807176[m Merge pull request #246 from tinystork/codex/mettre-à-jour-histogramme-après-traitement
[33mde0567a[m Refresh histogram after final stack
[33m59a0951[m Merge pull request #245 from tinystork/codex/ajouter-valeur-par-défaut-au-drizzle_fillval
[33m5485611[m Add default drizzle fill value
[33m5f2fca9[m Merge pull request #244 from tinystork/codex/modifier-logique-de-batching-dans-_worker
[33md41d4b8[m Handle incremental drizzle batches
[33m05f9159[m drizzle incremental
[33mfbee87a[m Merge pull request #243 from tinystork/codex/mettre-à-jour-comparaison-assert-et-tests-unitaires
[33mf30cd67[m Allow minor decrease in drizzle weight
[33m3b0cd18[m Merge pull request #242 from tinystork/codex/supprimer-vérification-conditionnelle-du-poids
[33m5250691[m Allow weight map override for drizzle batches
[33m49bd34f[m Merge pull request #241 from tinystork/codex/ajouter-des-assertions-et-des-tests-pour-drizzle
[33m34472ad[m Add wht checks and pixmap validation
[33m3b9aefc[m Merge pull request #240 from tinystork/codex/mettre-à-jour-les-instructions-ci-pour-installer-les-paquets
[33m158a8d5[m docs: clarify dev setup
[33mc6a08b7[m  travail sur inc drizzle
[33mc1f4047[m Merge pull request #239 from tinystork/codex/vérifier-out_wht-pour-valeurs-non-nulles
[33m29da23d[m Check incremental drizzle weights
[33m15f35db[m Merge pull request #238 from tinystork/codex/update-pixmap-computation-and-clipping
[33md2d23af[m Clip pixmap after origin1
[33mfb3dea9[m Merge pull request #237 from tinystork/codex/vérifier-et-ajuster-les-coordonnées-du-pixmap
[33m4a6bd86[m log pixmap ranges and adjust mapping
[33mbd3ac17[m Merge pull request #236 from tinystork/codex/mettre-à-jour-gestion-des-données-de-pluie
[33ma3ed32a[m Store incremental drizzle output
[33m90f1799[m Merge pull request #235 from tinystork/codex/refactor-drizzle-initialization-and-processing
[33m54515d6[m refactor: manage drizzle buffers internally
[33m7e1eab1[m Merge pull request #234 from tinystork/codex/refactor--_save_final_stack--for-incremental-drizzle
[33m3ba53bd[m Fix incremental drizzle final stack
[33m0805c4f[m Merge pull request #233 from tinystork/codex/afficher-le-reflet-du-fits-final-dans-la-prévisualisation
[33m302bf41[m Load final FITS for preview
[33me6e580a[m Revert "Merge pull request #231 from tinystork/codex/update-_save_final_stack-logic"
[33m618bcb4[m Merge pull request #232 from tinystork/codex/modifier-branche-fallback-dans-main_window.py
[33m4e89cad[m Use raw ADU fallback for final preview
[33m4cbc536[m Merge pull request #231 from tinystork/codex/update-_save_final_stack-logic
[33m15c61d2[m Use raw final stack for preview image
[33m31a9c62[m Merge pull request #230 from tinystork/codex/corriger-sauvegarde-fits-avec-adu-brut
[33me6b685b[m Fix FITS saving to use raw ADU data
[33m4dfaf46[m Merge pull request #229 from tinystork/4m89s3-codex/ajouter-paramètre-preserve_linear_output
[33m53bf274[m Merge branch 'graphic-Beta' into 4m89s3-codex/ajouter-paramètre-preserve_linear_output
[33m9c6406f[m Fix undefined preserve_linear_output_flag
[33mc6958bc[m Merge pull request #226 from tinystork/codex/créer-un-test-unitaire-pour-_save_final_stack
[33m471565a[m Merge branch 'graphic-Beta' into codex/créer-un-test-unitaire-pour-_save_final_stack
[33mb41a38b[m Add preserve_linear_output option and tests
[33m9231ebf[m Merge pull request #225 from tinystork/codex/mettre-à-jour-seestarqueuedstacker._save_final_stack
[33m5b519d5[m fix save_final_stack clipping
[33m770ba25[m Merge pull request #224 from tinystork/codex/ajouter-description-de-l-option-preserve-linear-output
[33me115c8d[m docs: describe Preserve Linear Output
[33m5273ee4[m Merge pull request #223 from tinystork/codex/modifier-_worker-pour-afficher-un-message-plate-solving
[33m35f9794[m Log plate-solving only when needed
[33m5687eac[m Merge pull request #222 from tinystork/codex/audit-print-statements-et-configurer-logging
[33mab55871[m Replace print statements with logging
[33m3729cdc[m Merge pull request #221 from tinystork/codex/consolider-attribut-reproject_between_batches
[33m4440b52[m Unify inter-batch reprojection flag
[33mf5951a3[m Merge pull request #220 from tinystork/codex/choisir-documentation-principale-et-fusionner
[33m5c1723e[m Consolidate documentation
[33m15b01d9[m Merge pull request #219 from tinystork/codex/créer-un-fichier-license
[33m66604ce[m Add GPLv3 license
[33m36d218d[m Merge pull request #218 from tinystork/codex/inspect-et-modifier-queue_manager.py
[33me467a56[m Remove unused grid computation helper
[33m3444709[m Merge pull request #217 from tinystork/codex/unify-attributes-related-to-reprojection
[33m880f1cf[m Unify inter-batch reprojection flag
[33mbd759cd[m Merge pull request #216 from tinystork/codex/corriger-état-option-inter-batch-reprojection
[33m35689e5[m Fix persistent checkbox for inter-batch reprojection
[33mb08d4bc[m Merge pull request #215 from tinystork/codex/corriger-l-argument-inattendu-dans-start_processing
[33m8a8547f[m fix gui start_processing compatibility
[33m842f8c8[m Merge pull request #214 from tinystork/codex/corriger-erreur-typeerror-dans-seestarqueuedstacker
[33m228a9a8[m Fix compatibility initializing queued stacker
[33m591192e[m Merge pull request #213 from tinystork/codex/corriger-régression-images-noires-et-reproject
[33ma8acece[m Fix percentile normalization for reproject stacks
[33maf051a9[m Merge pull request #212 from tinystork/codex/adapter-traitement-image-pour-éviter-sortie-noire
[33mf8a1f9e[m Fix classic stacking output calculation
[33m52561d8[m Merge pull request #211 from tinystork/codex/ajouter-déclaration-de-débogage-et-tests
[33m166a8d8[m Add debug print and test for cumulative stack counter
[33mda40d41[m Merge pull request #210 from tinystork/5bhy3l-codex/corriger-le-bug-de-traitement-d-images
[33m3994c13[m Fix memmap accumulation shape checks
[33m472eeb6[m Merge pull request #209 from tinystork/codex/corriger-le-bug-de-traitement-d-images
[33m1e4bb21[m Fix memmap accumulation shape checks
[33mc60f56c[m Merge pull request #208 from tinystork/codex/ajouter-appels-update_progress-et-documenter-comportements
[33m6b95b78[m Log skipped batches and zero coverage
[33m5ca9945[m Merge pull request #207 from tinystork/codex/instrumenter-et-valider-le-traitement-du-batch
[33m7978e3d[m instrument combine_batch_result
[33mc0b3cd8[m Merge pull request #206 from tinystork/codex/update-progress-in-_combine_batch_result
[33m867e2e5[m log skipped batches
[33m63d04ae[m Merge pull request #205 from tinystork/6sd00i-codex/ajouter-des-logs-de-débogage-dans-queue_manager.py
[33mae58a8e[m Merge branch 'graphic-Beta' into 6sd00i-codex/ajouter-des-logs-de-débogage-dans-queue_manager.py
[33m3e5252f[m Add debug prints in _combine_batch_result
[33m2c4c405[m Merge pull request #204 from tinystork/codex/ajouter-des-logs-de-débogage-dans-queue_manager.py
[33m1ca230f[m Merge branch 'graphic-Beta' into codex/ajouter-des-logs-de-débogage-dans-queue_manager.py
[33m9044437[m Add debug prints in _combine_batch_result
[33m20376c4[m Merge pull request #203 from tinystork/codex/add-debug-statement,-confirm-assignments,-write-unit-test
[33m187d006[m Add debug counter log and stack counter test
[33m988865a[m Merge pull request #202 from tinystork/codex/vérifier-mise-à-jour-des-memmaps-et-ajout-de-logs
[33mde67207[m Add memmap debug logging
[33m17b6827[m Merge pull request #201 from tinystork/codex/ajouter-psutil-aux-tests
[33m873c688[m Add psutil to test requirements and document
[33m3b58e80[m Merge pull request #200 from tinystork/codex/log-min/max-of-cumulative-memmaps
[33m3c26636[m Log memmap stats after batch
[33mc724ef8[m Merge pull request #199 from tinystork/codex/ajouter-un-bloc-try/except-pour-psutil
[33m0de2c45[m Make psutil optional
[33mff27a00[m requirements
[33m4d768fa[m Merge pull request #198 from tinystork/codex/fix-valid-pixel-mask-when-luminance-faible
[33m84a9c87[m Fix valid pixel mask when luminance nearly zero to avoid zero-weight stacking
[33m2af73be[m Merge pull request #197 from tinystork/codex/corriger-image-noire-après-traitement
[33m402ce53[m Reduce variance rejection threshold
[33m91a4405[m Merge pull request #196 from tinystork/codex/vérifier-pipeline-de-sauvegarde-image
[33m7f38c6a[m Fix constant image normalization
[33md6e4be4[m Merge pull request #195 from tinystork/oprsdv-codex/passer-le-flag--enable-inter-batch-reprojection--au-backend
[33m5ea57dc[m Merge branch 'graphic-Beta' into oprsdv-codex/passer-le-flag--enable-inter-batch-reprojection--au-backend
[33m8cfd5eb[m Use reproject_and_coadd for inter-batch classic reprojection
[33m0d0cd05[m Merge pull request #194 from tinystork/codex/passer-le-flag--enable-inter-batch-reprojection--au-backend
[33mb40a6f0[m Pass inter batch reprojection flag to backend
[33meab26f8[m Merge pull request #192 from tinystork/lyee85-codex/corriger-indexations-et-appliquer-carte-wht
[33m6664138[m Fix classic stacking reprojection
[33m0eaf373[m Merge pull request #191 from tinystork/ewwhrz-codex/corriger-indexations-et-appliquer-carte-wht
[33m0bc18d1[m Fix classic stacking reprojection
[33m99ec889[m Merge pull request #189 from tinystork/rl3r7b-codex/activer-reprojection-inter-batch-en-mode-classic
[33m1a5b774[m Merge branch 'graphic-Beta' into rl3r7b-codex/activer-reprojection-inter-batch-en-mode-classic
[33mbb15d18[m Fix inter-batch reprojection handling
[33m5dacc32[m Merge pull request #188 from tinystork/codex/activer-reprojection-inter-batch-en-mode-classic
[33m7b687c3[m feat: enable classic inter-batch reprojection
[33ma539bba[m Merge pull request #187 from tinystork/codex/corriger-appel-astap-pour-traitement-classique
[33m3887cee[m enable WCS solving when interbatch reprojection
[33mb684577[m Merge pull request #186 from tinystork/jur4qu-codex/comparer-échec-appel-astap-avec-zemosaic
[33m6e2c2d5[m Fix ASTAP stack solve by using grayscale
[33m0a0b487[m Merge pull request #185 from tinystork/zxm4bi-codex/comparer-échec-appel-astap-avec-zemosaic
[33me38b303[m Fix AstrometrySolver header update method
[33m309b39c[m Merge pull request #184 from tinystork/codex/mettre-à-jour-readme-avec-section-tests
[33m9da8825[m docs: add running tests section
[33m0ad38de[m Merge pull request #183 from tinystork/codex/ajouter-fallback-wcs-avec-avertissement
[33m7a8599a[m Add WCS fallback during batch processing
[33m167d57e[m Merge pull request #182 from tinystork/codex/comparer-échec-appel-astap-avec-zemosaic
[33m815e2c9[m Fix ASTAP header update helper
[33me1e8fc3[m Merge pull request #181 from tinystork/gr00c8-codex/reprojet-désactivé-en-stacking-classique
[33m1455fe0[m fix settings sync for interbatch reprojection
[33mbedd4f1[m Merge pull request #180 from tinystork/dx5uxi-codex/reprojet-désactivé-en-stacking-classique
[33m0bfa318[m Enable reference WCS solving when inter-batch reprojection is on
[33m0c34a4c[m Merge pull request #179 from tinystork/codex/reprojet-désactivé-en-stacking-classique
[33mbdf4a12[m Enable reference WCS solving when inter-batch reprojection is on
[33mf605ef6[m Merge pull request #178 from tinystork/codex/refactor-print-statements-and-add-logging
[33m89cea11[m Use logging instead of print during import
[33m59fe63c[m Merge pull request #177 from tinystork/codex/renommer-et-mettre-à-jour-option
[33m3061622[m refactor: unify batch reprojection setting
[33mabcbf82[m Merge pull request #176 from tinystork/codex/implement-reproject_to_reference-method-and-update-processin
[33me7f2dac[m Add inter-batch reprojection flag and tests
[33mdd3b7b8[m Merge pull request #175 from tinystork/codex/run-flake8-and-corriger-les-erreurs-majeures
[33md4cfc90[m Fix flake8 spacing and line length issues
[33md8e6418[m Merge pull request #174 from tinystork/codex/étendre-assemble_final_mosaic_with_reproject_coadd-et-implém
[33m4915c7b[m Add crop re-solve and grid preparation
[33m0d26633[m Merge pull request #173 from tinystork/codex/ajouter-tests-pour-reproject_utils
[33m2613362[m Add tests for reproject utilities
[33mc767dc1[m Merge pull request #172 from tinystork/codex/créer-module-reproject_utils-avec-stubs
[33mea56ae9[m Add reproject utility module and update imports
[33ma0cdb40[m Merge pull request #171 from tinystork/6yoypo-codex/vérifier-l-utilisation-de-reproject-entre-les-batches
[33mf98aec7[m Merge branch 'graphic-Beta' into 6yoypo-codex/vérifier-l-utilisation-de-reproject-entre-les-batches
[33m3f17693[m Merge pull request #169 from tinystork/gqdpv0-codex/vérifier-l-utilisation-de-reproject-entre-les-batches
[33mab8d229[m Merge branch 'graphic-Beta' into gqdpv0-codex/vérifier-l-utilisation-de-reproject-entre-les-batches
[33mdc65b30[m Add GUI logs for batch WCS solving and skipped reproject
[33m2b106f7[m Add GUI logs for batch WCS solving and skipped reproject
[33mbac2c0e[m Merge pull request #168 from tinystork/6yoypo-codex/vérifier-l-utilisation-de-reproject-entre-les-batches
[33m9791c3d[m Add GUI logs for winsor step
[33m7f132c1[m Merge pull request #167 from tinystork/codex/vérifier-l-utilisation-de-reproject-entre-les-batches
[33m11db07e[m Add GUI logs for entering and exiting reprojection
[33mf235b2e[m Merge pull request #166 from tinystork/r0wr29-codex/ajouter-reprojection-inter-batch-avec-wcs
[33m93188fd[m Merge branch 'graphic-Beta' into r0wr29-codex/ajouter-reprojection-inter-batch-avec-wcs
[33m647b003[m Rename inter-batch reprojection setting
[33mf67e956[m Merge pull request #165 from tinystork/codex/étendre-la-docstring-de-solve_image_wcs
[33m3a5b91f[m Clarify solve_image_wcs behavior
[33m93bd5e8[m Merge pull request #164 from tinystork/codex/ajouter-une-nouvelle-ligne-à-la-fin-du-fichier
[33me34756a[m Add newline after EOF comments
[33m646f003[m Merge pull request #163 from tinystork/codex/ajouter-log-avant-if-dans-_combine_batch_result
[33me616b4b[m Merge branch 'graphic-Beta' into codex/ajouter-log-avant-if-dans-_combine_batch_result
[33m270854a[m Add interbatch reprojection debug log
[33mf7c64ee[m Merge pull request #162 from tinystork/codex/renommer-enable_reprojection_between_batches
[33m4ef0cbc[m Rename interbatch reprojection setting
[33m10ac4e8[m Merge pull request #161 from tinystork ajouter-reprojection-inter-batch-avec-wcs
[33m5fef564[m Add inter-batch plate solving and reprojection
[33md8ce015[m Merge pull request #160 from tinystork ajouter-zoom-et-réinitialisation-histogramme
[33mfdb7f0d[m Merge branch 'graphic-Beta' into liinql-codex/ajouter-zoom-et-réinitialisation-histogramme
[33md232d0b[m Reposition histogram controls in toolbar
[33m5fad16c[m Merge pull request #159 from tinystork ajouter-zoom-et-réinitialisation-histogramme
[33mfc529df[m Add histogram zoom controls
[33mae0eacc[m Merge pull request #158 from tinystork/fbptrb-codex/traduire-les-textes-dans-local_solver_gui.py
[33mb9a8f81[m Merge branch 'graphic-Beta' into fbptrb-codex/traduire-les-textes-dans-local_solver_gui.py
[33m7d02532[m Fix translation helper recursion
[33mf33a40d[m Merge pull request #157 from tinystork traduire-les-textes-dans-local_solver_gui.py
[33m36ba065[m local solver gui i18n
[33m61c7c51[m Merge pull request #156 from tinystork modifier-gestion-erreurs-label-remaining
[33mb0607bc[m Update remaining files display
[33mf04434e[m Merge pull request #155 from tinystork ajouter-des-logs-détaillés-pour-les-étapes-clés
[33m4034052[m Add verbose logging for solver, reprojection and winsor
[33m650cdfe[m Merge pull request #154 from tinystork ajouter-reprojection-entre-batchs
[33m8aff193[m Add optional batch reprojection
[33m86c7595[m Merge pull request #153 from tinystork mettre-à-jour-fenêtre-de-configuration-du-solveur
[33mc62c112[m Merge branch 'graphic-Beta' into r9pmm9-codex/mettre-à-jour-fenêtre-de-configuration-du-solveur
[33mcb29160[m Extend solver settings
[33mb071d55[m Merge pull request #152 from tinystork mettre-à-jour-fenêtre-de-configuration-du-solveur
[33me45ab3c[m fix(gui): clean solver config layout
[33me8371e6[m correction pour astrometry
[33m5ecda0c[m Merge pull request #151 from tinystork/codex/fix-lancement-mosaïque-définitif
[33mb73bf92[m Fix solver method key
[33m3b8ff52[m Merge pull request #150 from tinystork/codex/déplacer-et-configurer-sous-cadre-astap
[33m13523f7[m Move ASTAP config next to solver and update fallback
[33m77e4074[m Merge pull request #149 from tinystork/codex/mettre-à-jour-logique-de-solver-pour-astrometry-et-astap
[33mc96942d[m Fix solver selection handling
[33m7895f01[m Merge pull request #148 from tinystork/codex/ajouter-un-fallback-pour-astrometry
[33m1e7294b[m Add astrometry fallbacks and config
[33mf330c0c[m VERSION 3.0.0
[33m9b05834[m Merge pull request #147 from tinystork/codex/remplacer-logique-_browse_astrometry_local_path
[33m166f130[m Update astrometry path browser
[33m8066a50[m Merge pull request #146 from tinystork/codex/mettre-à-jour-le-binding-solver_combo
[33m06b9852[m Call solver change handler after combobox refresh
[33mdaf1a95[m Merge pull request #145 from tinystork/9yb4z9-codex/ajouter-configuration-astrometry-dans-l-ui
[33mdaa2158[m Merge branch 'graphic-Beta' into 9yb4z9-codex/ajouter-configuration-astrometry-dans-l-ui
[33m3cd99a6[m Fix zemosaic_config import
[33m8a872ac[m Merge pull request #144 from tinystork/codex/ajouter-configuration-astrometry-dans-l-ui
[33m322239a[m Fix worker import and remove duplicate config keys
[33maaeccef[m Merge pull request #143 from tinystork/codex/collect-solver-settings-from-gui
[33m2515ffb[m pass solver settings dict
[33m9d68364[m Merge pull request #142 from tinystork/codex/ajouter-des-entrées-à-default_config
[33m1ef2c37[m Add solver config options
[33m2362341[m Merge pull request #141 from tinystork/codex/extend-config-and-update-gui-with-solver-settings
[33mc8f681c[m Add solver configuration options
[33md69d9b0[m Merge pull request #140 from tinystork/codex/résoudre-l-erreur-ioregistryerror
[33mb003cb8[m Fix classical stacking with CCDData
[33m72f952c[m Merge pull request #139 from tinystork/codex/exposer-paramètres-astrométriques-dans-mosaicsettingswindow
[33m0ecfd73[m Add getters for astrometry config in mosaic window
[33me39efb9[m Merge pull request #138 from tinystork/codex/déplacer-widgets-de-configuration-astrométriques-vers-mosaic
[33m09eb083[m Move solver configuration UI to mosaic window
[33mdc16960[m version 2.9.0 zemosaic added
[33mea474a6[m Merge pull request #137 from tinystork/codex/ajouter-vérification-config-dans-localsolversettingswindow
[33m1ee5c81[m Add fallback config loading in LocalSolverSettingsWindow
[33m050b515[m Merge pull request #136 from tinystork/codex/corriger-erreur-de-création-fenêtre-paramètres-solveurs
[33mf94873c[m Fix missing config attribute
[33m34bc4af[m reset
[33m8968be1[m Merge pull request #124 from tinystork supprimer-option-et-déplacer-paramètres-astap
[33mdb6a15a[m Move ASTAP settings to solver config window and remove RA/DEC hints
[33m9479604[m Merge pull request #123 from tinystork modifier-bouton-mosaic-pour-ouvrir-run_zemosaic.py
[33md442e2f[m fix: launch ZeMosaic via script path
[33m58b4432[m Merge pull request #121 from tinystork améliorer-la-cohérence-wcs-post-re-solve
[33m127457f[m Improve ASTAP solve wrapper with fallback attempts
[33md52dc87[m Merge pull request #120 from tinystork optimize-mosaic-grid-calculation
[33m67998bc[m Update mosaic worker grid preparation
[33m5a63a2c[m Merge pull request #119 from tinystork update-default-use_radec_hints-flag
[33mf6407c4[m Set RA/DEC hints default
[33mbaab5a0[m Merge pull request #118 from tinystork ajouter-section-readme-pour-packages-de-tests
[33m3667501[m docs: list test dependencies
[33mfcd02c5[m Merge pull request #117 from tinystork add-use_radec_hints-parameter-to-solver
[33m84a118c[m Add use_radec_hints option
[33m53e4252[m Merge pull request #116 from tinystork ajouter-un-log-ra/dec-avant-astap
[33m0edc8ee[m log RADEC hints before executing astap
[33m205b6db[m Merge pull request #115 from tinystork étendre-readme.md-pour-ra/dec-et-pixel-scale
[33mfa39260[m docs: explain solver hints
[33m25d68e0[m Merge pull request #114 from tinystork implémenter-un-helper-pour-scale-pixel
[33m98a8916[m Merge branch 'graphic-Beta' into codex/implémenter-un-helper-pour-scale-pixel
[33m811fe99[m feat(astap): derive pixel scale
[33mb3cbf36[m Merge pull request #113 from tinystork investiguer-suppression-ou-renommage-de-zemosaic_astrometry
[33m3d407d3[m feat: reintroduce astap wrapper
[33m2447aad[m Merge pull request #112 from tinystork modifier-fonction-assemble_final_mosaic_incremental
[33m8e29d1c[m Add WCS re-solve option for incremental mosaic
[33mef2323f[m Merge pull request #111 from tinystork mettre-à-jour-readme-pour-use_radec_hints
[33mee72ec8[m docs: clarify RADEC hints
[33m78183b2[m Merge pull request #110 from tinystork mise-à-jour-des-tests-pour-ra/dec
[33m05c6781[m test: check radec hints toggle
[33mac154c3[m Merge pull request #109 from tinystork modifier-le-comportement-du-solver-avec-use_radec_hints
[33ma1e9da6[m Respect use_radec_hints when re-solving cropped tiles
[33m2e841dd[m Merge pull request #108 from tinystork modifier-valeur-par-défaut-use_radec_hints
[33m413f072[m Change default RA/DEC hints
[33m20fa8ea[m Merge pull request #107 from tinystork supprimer-commentaires-obsolètes-et-imports-inutiles
[33maca761a[m Clean comments and imports
[33m064834a[m Merge pull request #106 from tinystork modifier-l-en-tête-avant-writeto
[33m9ba5ab7[m Clean solver temp header
[33m1bd163d[m Merge pull request #105 from tinystork ajouter-option-use_radec_hints
[33m679b901[m feat: allow disabling astap radec hints
[33m5cd9824[m Merge pull request #104 from tinystork mettre-à-jour-le-dérivé-ra/dec-et-les-tests
[33mb22561e[m fix wcs solver hints after crop
[33me256490[m Merge pull request #103 from tinystork modifier-fallback-astap_search_radius
[33m94f08d2[m Use configurable default ASTAP search radius
[33m7a930b8[m Merge pull request #102 from tinystork identifier-problème-résolution-astap
[33mc5dd1da[m Format ASTAP search radius to two decimals
[33m4d8e671[m Merge pull request #101 from tinystork modifier-la-fonction-_parse_wcs_file_content
[33m79d9324[m Handle non-standard ASTAP WCS files with relax=True
[33m3b65e85[m Merge pull request #100 from tinystork résoudre-problèmes-astrométriques-avec-astap
[33m271c7d5[m Allow configuring ASTAP downsample and sensitivity
[33me279983[m Merge pull request #99 from tinystork add-pixel-scale-comparison-and-unit-test
[33m3190638[m Check pixel scale after optimal grid
[33m133a7ba[m Merge pull request #98 from tinystork mettre-à-jour-exemples-readme-et-scripts-cli
[33m9a82eb6[m docs: document solver_settings usage
[33mb0bba22[m Merge pull request #97 from tinystork mettre-à-jour-le-readme-avec-une-section--development-setup
[33m5d33e83[m Merge branch 'graphic-Beta' into syiogz-codex/mettre-à-jour-le-readme-avec-une-section--development-setup
[33m20b76b1[m Refine development setup section
[33mdb8e136[m Merge pull request #96 from tinystork mettre-à-jour-le-readme-avec-une-section--development-setup
[33m92d9bce[m docs: add development setup section
[33mbfb51cf[m Merge pull request #95 from tinystork analyser-échec-de-solution-astrométrique-astap
[33m42eea63[m Avoid nested unaligned folders
[33m418e1be[m Merge pull request #94 from tinystork corriger-le-fonctionnement-de-zemosaic
[33md4659b8[m Add CLI mode and config paths for ZeMosaic
[33m6806e48[m Merge pull request #93 from tinystork utiliser-la-solution-astrométrique-zeseestarstacker
[33me7bb35b[m Use sidecar WCS files when available
[33m04e1172[m Merge pull request #92 from tinystork corriger-erreur-f-string-dans-astrometry_solver.py
[33md346cae[m Fix f-string quoting in astrometry_solver
[33m38bc225[m Merge pull request #91 from tinystork ajouter-un-test-dans-test_mosaic_worker.py
[33m4a27444[m Merge branch 'graphic-Beta' into codex/ajouter-un-test-dans-test_mosaic_worker.py
[33m66cdf29[m Add test for solver call when tile lacks WCS
[33m45184d7[m Merge pull request #90 from tinystork ajouter-gestion-des-erreurs-et-test-de-régression
[33mcbd6744[m Handle stretch failures and add regression test
[33m8ad4ed5[m Merge pull request #89 from tinystork créer-script-run_mosaic.py
[33mf7bc519[m Add CLI script for hierarchical mosaic
[33mb62940a[m Merge pull request #88 from tinystork refactor-imports-in-alignment.py
[33m606edd9[m Remove duplicate imports
[33md79ac0e[m Merge pull request #87 from tinystork remplacer-print-par-logging
[33mdfebbea[m Clean debug logging
[33m8dc3611[m Merge pull request #86 from tinystork ajouter-routine-d-alignement-dans-assemble_final_mosaic
[33m2f0bdd2[m feat: align tiles before mosaic assembly
[33m744a6fd[m Merge pull request #85 from tinystork résoudre-problème-avec-tuiles-rognées
[33m2e8349a[m Add RA/DEC hints to ASTAP command
[33m09222a4[m Merge pull request #84 from tinystork résoudre-problème-avec-tuiles-rognées
[33mc9a1c51[m preserve RA DEC when solving cropped tiles
[33m543749c[m Merge pull request #83 from tinystork add-re-solve-cropped-tiles-feature
[33mbea2c9e[m Add re-solve cropped tiles option
[33mcbd865c[m Merge pull request #82 from tinystork modifier-appel-à-stretch_auto_asifits_like
[33m9ac6c60[m Use asinh_a param in preview stretch
[33m56c61d4[m Merge pull request #81 from tinystork/ utiliser-reproject-pour-créer-le-mosaic-final
[33m57af218[m Add optional WCS re-solve after cropping
[33m2aa444d[m Merge pull request #80 from tinystork set-pixel_shape-and-add-error-handling
[33mec75627[m Set pixel_shape after tile load and crop
[33m1db607a[m Merge pull request #79 from tinystork deduplicate-list-and-log-crval
[33mca34447[m Deduplicate master tiles before assembly
[33m2ab7fa2[m Merge pull request #78 from tinystork étendre-la-gestion-de-solver_settings
[33mba83dcd[m Handle astrometry_method setting
[33mf04ffa1[m Merge pull request #77 from tinystork améliorer-la-logique-d-importation-et-gestion-de-wcs
[33m4a35cf9[m Improve fallback for optimal wcs
[33m9c978e5[m Merge pull request #76 from tinystork modifier-assemble_final_mosaic-incrementale-et-coadd
[33mf08a135[m Update cropping logic
[33mb80f7c0[m Merge pull request #75 from tinystork corriger-erreur-module-shapely-manquant
[33mdb26282[m Handle missing shapely in mosaic worker
[33m0576740[m Merge pull request #74 from tinystork realigner-les-master_tiles-après-le-rognage
[33mf1fbe5f[m fix: realign master tiles after cropping
[33mad53222[m Merge pull request #73 from tinystork remplacer-fonction-find_optimal_celestial_wcs
[33m7f4c786[m Add fallback mosaic grid computation
[33m222656b[m Merge pull request #72 from tinystork fix-wcs-grid-calculation-for-mosaic
[33m218e7a0[m Fix mosaic grid calculation using optimal WCS
[33m2b452b2[m Merge pull request #71 from tinystork ajouter-la-mise-à-jour-des-comboboxes-pour-les-langues
[33m1e87d10[m Refresh combobox texts on UI build
[33m3b3aeee[m Merge pull request #70 from tinystork remplacer-print-par-logging-et-ajouter-mode-verbose
[33m43ab09b[m Add verbose logging toggle and update docs
[33m112df37[m Merge pull request #69 from tinystork mettre-à-jour-phase-4-dans-zemosaic_worker.py
[33mf9c5e0d[m Use dynamic WCS grid calculation
[33m2679baf[m Merge pull request #68 from tinystork ajouter-fonction-de-calcul-wcs-dynamique
[33m65a214e[m Add dynamic WCS grid computation
[33m9989204[m Merge pull request #67 from tinystork afficher-erreur-pour-reproject-manquant
[33m5282880[m Improve reproject guidance
[33mb33b76e[m Merge pull request #66 from tinystork vérifier-propagation-des-variables-zemosaic
[33m35a00b4[m Log solver config
[33md48e412[m Merge pull request #65 from tinystork modifier-run_hierarchical_mosaic-pour-passer-astrometry_solv
[33mc5340c0[m Pass AstrometrySolver through pipeline
[33m9ff8256[m Merge pull request #64 from tinystork remplacer-print-par-logging-dans-run_zemosaic.py
[33meb3689d[m Replace prints with logging in run_zemosaic
[33m8d99061[m Merge pull request #63 from tinystork remplacer-print-par-logger-et-ajouter-flag-de-verbosité
[33m4c9e46c[m Replace prints with logger and add verbose flag
[33m4055034[m Merge pull request #62 from tinystork extend-run_zemosaic-to-support-local_solver_preference
[33md9535ff[m pass solver preference to ZeMosaic
[33me4e3bde[m Merge pull request #61 from tinystork forward-solver-settings-via-command-line-arguments
[33mab19377[m Forward solver settings to ZeMosaic
[33m4e13829[m Merge pull request #60 from tinystork refactor-imports-in-mosaic_processor.py
[33m0df1c89[m Use TYPE_CHECKING for optional queue manager import
[33m80e6f5a[m Merge pull request #59 from tinystork refactor-imports-in-queue_manager.py
[33m3bfbddd[m Refactor local import for create_master_tile
[33m7e41498[m Merge pull request #58 from tinystork wrap-psutil-import-in-try/except-block
[33m070df14[m Make psutil optional
[33m8db176d[m Merge pull request #57 from tinystork supprimer-champs-astap-de-la-config
[33md412513[m Remove ASTAP paths from config
[33m0702787[m Merge pull request #56 from tinystork supprimer-configuration-astap-et-nettoyer-variables
[33m12d4df2[m Remove ASTAP config UI
[33ma15fdcd[m Merge pull request #55 from tinystork supprimer-zemosaic_astrometry.py-et-nettoyer-le-code
[33mc2bbded[m Remove unused zemosaic_astrometry module
[33md7fd077[m Merge pull request #54 from tinystork modifier-signature-de-run_hierarchical_mosaic
[33meaf7d25[m Merge branch 'graphic-Beta' into codex/modifier-signature-de-run_hierarchical_mosaic
[33m04f5859[m Refactor solver settings usage
[33m60e91b5[m Merge pull request #53 from tinystork intégrer-astrometrysolver-dans-zemosaic_worker.py
[33m2ac5586[m Switch to AstrometrySolver
[33m84b3b3c[m Merge pull request #52 from tinystork charger-traductions-autonomes-de-zemosaic
[33mb9779a1[m fix zemosaic localization import
[33m58cd3b6[m Merge pull request #51 from tinystork lancer-run_mosaic.py-depuis-main_window
[33m27e7009[m Merge branch 'graphic-Beta' into 5mxnnu-codex/lancer-run_mosaic.py-depuis-main_window
[33mc2902c6[m Fix ZeMosaic launcher to work from any directory
[33mec34ff1[m Merge pull request #50 from tinystork mettre-à-jour-run_zemosaic-pour-définir-le-répertoire-de-tra
[33m03b97c4[m Fix run_zemosaic launch path
[33m54609ae[m Merge pull request #49 from tinystork lancer-run_mosaic.py-depuis-main_window
[33m17fd706[m Launch ZeMosaic from main GUI
[33m7e21311[m Merge pull request #48 from tinystork corriger-erreur-de-syntaxe-try-except
[33m792be10[m Fix stray try statement in mosaic settings window
[33ma1a919a[m Update run_zemosaic to launch external GUI
[33md151386[m Merge pull request #44 from tinystork mettre-à-jour-local_solver_button_text
[33m7d2e17b[m[33m ([m[1;31morigin/codex/mettre-à-jour-local_solver_button_text[m[33m)[m Update solver button translation
[33m17d9f12[m Merge pull request #43 from tinystork refactor-api-key-handling
[33mbdc8478[m Remove API key entry from mosaic window and add to solver window
[33m734fc36[m Merge pull request #42 from tinystork ajouter-eta-plus-précise-après-validation
[33m3d9f55a[m[33m ([m[1;31morigin/codex/ajouter-eta-plus-précise-après-validation[m[33m)[m Add ETA updates for stacking batches
[33mbf2e352[m Merge pull request #41 from tinystork add-eta-and-logging-for-stacking-process
[33ma6f4b0f[m Add ETA and progress logging during master tile stacking
[33me4c5f8b[m Merge pull request #40 from tinystork localiser-étiquettes-et-bouton-dans-l-interface-zeseestarsta
[33m2e21318[m[33m ([m[1;31morigin/codex/localiser-étiquettes-et-bouton-dans-l-interface-zeseestarsta[m[33m)[m Localize status label and SNR rejection button
[33mf4ab02c[m Merge pull request #39 from tinystork mettre-à-jour-le-gestionnaire-d-erreurs-eta
[33ma708e72[m[33m ([m[1;31morigin/codex/mettre-à-jour-le-gestionnaire-d-erreurs-eta[m[33m)[m Improve ETA error handling
[33mc053cc5[m Merge pull request #38 from tinystork add-auto-stretch-and-zoom-buttons
[33m07d85d1[m feat: add preview auto-stretch and zoom controls
[33m3f6281f[m Merge pull request #37 from tinystork mettre-à-jour-remaining_time_var-dans-progressmanager
[33md6248cc[m[33m ([m[1;31morigin/codex/mettre-à-jour-remaining_time_var-dans-progressmanager[m[33m)[m feat(gui): sync ETA to progress manager
[33m5ed08fd[m Merge pull request #36 from tinystork implémenter-moyenne-mobile-exponentielle-pour-eta
[33m5a38539[m[33m ([m[1;31morigin/codex/implémenter-moyenne-mobile-exponentielle-pour-eta[m[33m)[m Add smoothed ETA calculation
[33md62142d[m Merge pull request #35 from tinystork ajouter-variables-d-affichage-et-mises-à-jour-ui
[33m72d75a7[m[33m ([m[1;31morigin/codex/ajouter-variables-d-affichage-et-mises-à-jour-ui[m[33m)[m Add localized display vars for stacking options
[33md986d69[m Merge pull request #34 from tinystork add-translations-and-update-widgets_to_translate
[33m57cedf4[m Add missing localization keys for stacking methods
[33m1b5cb40[m Merge pull request #33 from tinystork localiser-les-étiquettes-de-méthode
[33ma06e1a8[m localize stacking method combo
[33mbe08b6d[m Merge pull request #32 from tinystork normaliser-les-cartes-de-poids-dans-les-fichiers-spécifiés
[33m848c655[m[33m ([m[1;31morigin/codex/normaliser-les-cartes-de-poids-dans-les-fichiers-spécifiés[m[33m)[m Normalize weight maps
[33m415e427[m Merge pull request #31 from tinystork replace-hardcoded-strings-with-self.tr-calls
[33m421aca9[m localize browse and scnr labels
[33m3aa5365[m Merge pull request #30 from tinystork modifier-gestion-de-pile-dans-queue_manager.py
[33mb205514[m Fallback to classic stacking when WCS missing
[33m684e481[m Merge pull request #29 from tinystork/codex/corriger-les-images-blanches-après-fusion
[33mca5b230[m[33m ([m[1;31morigin/codex/corriger-les-images-blanches-après-fusion[m[33m)[m Fix stretch percentile on flat images
[33m85a86ff[m Merge pull request #28 from tinystork refactor-imports-with-try/except-fallback
[33m47255e3[m Use relative imports with fallbacks
[33mbb14278[m Merge pull request #27 from tinystork fusionner-options-de-stacking
[33m3bdc898[m Merge stacking method and rejection algorithm
[33mc63a631[m Merge pull request #26 from tinystork supprimer-le-bloc-min/max-dans-_stack_batch
[33mc1589b5[m Remove batch min-max scaling and update docs
[33m5b4d31a[m Merge pull request #25 from tinystork refactor-widgets-in-main-window
[33m0af02e3[m Add unified stacking method option
[33m0c8a416[m Merge pull request #24 from tinystork mettre-à-jour-la-docstring-de-_stack_batch
[33m1809679[m[33m ([m[1;31morigin/codex/mettre-à-jour-la-docstring-de-_stack_batch[m[33m)[m Update stack_batch docstring to mention ZeMosaic
[33m60febe0[m Merge pull request #23 from tinystork refactor-_stack_batch-to-update-header
[33m800dc8c[m Simplify stack batch header update
[33mac4f784[m Merge pull request #22 from tinystork ajouter-import-tempfile-sans-doublon
[33ma361dc4[m Add missing tempfile import
[33m1be9dd5[m Merge branch 'graphic-Beta' of https://github.com/tinystork/zeseestarstacker into graphic-Beta
[33mccd7e39[m Merge pull request #21 from tinystork implement-create_master_tile-in-queue_manager.py
[33m5f397e4[m Use create_master_tile for batch stacking
[33m0d328c0[m Merge branch 'graphic-Beta' of https://github.com/tinystork/zeseestarstacker into graphic-Beta
[33m56de4c3[m Merge pull request #20 from tinystork ajouter-des-traductions-et-mettre-à-jour-widgets
[33m11a3513[m[33m ([m[1;31morigin/codex/ajouter-des-traductions-et-mettre-à-jour-widgets[m[33m)[m Add stacking option translations and hook up labels
[33m3d42e57[m Merge branch 'graphic-Beta' of https://github.com/tinystork/zeseestarstacker into graphic-Beta
[33m37b4490[m Merge pull request #19 from tinystork étendre-les-méthodes-de-settingsmanager
[33mb752ca8[m[33m ([m[1;31morigin/codex/étendre-les-méthodes-de-settingsmanager[m[33m)[m Add advanced stacking options to SettingsManager
[33m9a87b53[m Merge branch 'graphic-Beta' of https://github.com/tinystork/zeseestarstacker into graphic-Beta
[33m5a5f510[m Merge pull request #18 from tinystork ajouter-widgets-dans-main_window.py
[33m8d5f248[m Add advanced stacking option controls
[33mbc8e326[m Merge branch 'main' of https://github.com/tinystork/zeseestarstacker into graphic-Beta
[33m7647b4b[m Merge pull request #17 from tinystork consolider-import-proj_plane_pixel_scales
[33m3c51968[m Remove duplicate proj_plane_pixel_scales imports
[33m9d4dd38[m Merge pull request #16 from tinystork add-linear_fit-and-sky_mean-options
[33m527b14a[m Add new normalization and weighting utilities
[33mc9e9ec8[m addition of zemosaic
[33m6d9b282[m Merge pull request #15 from tinystork décocher-par-défaut-la-case--enable-trail-detection
[33m01aad5e[m[33m ([m[1;31morigin/codex/décocher-par-défaut-la-case--enable-trail-detection[m[33m)[m Make trail detection disabled by default
[33m3651ecf[m comment clean up
[33m9ba10a8[m Merge branch 'graphic-Beta' of https://github.com/tinystork/zeseestarstacker into graphic-Beta
[33maae4e5f[m Merge pull request #14 from tinystork add-send-reference-button-and-functionality
[33me8fbfa6[m Add best reference communication
[33m663e29d[m Merge branch 'graphic-Beta' of https://github.com/tinystork/zeseestarstacker into graphic-Beta
[33ma6fc9e8[m Merge pull request #13 from tinystork extend-command-line-parser-with-language-options
[33mac083a4[m Add language options to analyzer
[33m61e500b[m Merge branch 'graphic-Beta' of https://github.com/tinystork/zeseestarstacker into graphic-Beta
[33mc609bc5[m Merge branch 'main' of https://github.com/tinystork/zeseestarstacker into graphic-Beta
[33m5a91a9c[m Merge pull request #12 from tinystork ajouter-option-de-nom-de-fichier-de-sortie
[33mbe7653c[m Add output filename setting
[33mefaf41f[m Merge pull request #11 from tinystork
[33m923fe53[m[33m ([m[1;31morigin/v6nurv-codex/trouver-et-corriger-un-bug[m[33m)[m Fix CUDA denoising method calls
[33mf64de95[m Merge pull request #10
[33ma1ba648[m[33m ([m[1;31morigin/version-2-9-9[m[33m, [m[1;32mversion-2-9-9[m[33m)[m Fix CUDA denoising fallback condition
[33md83f778[m Merge branch 'graphic-Beta'
[33m5a0dc46[m bug fix on mosaic
[33m2332d3e[m Merge branch 'graphic-Beta'
[33m7cc5f71[m version 2.6.6 fixes missing memap folder in mosaic mode and preview display and exit
[33m77b109c[m Merge branch 'graphic-Beta'
[33m77f60bb[m version 2.6.6 fixes missing memap folder in mosaic mode and preview display
[33mcec18f0[m Merge branch 'graphic-Beta'
[33m803ff2c[m version 2.6.3
[33m51ebd61[m version 2.6.3
[33m0bca5ca[m version 2.6.3
[33m0c55242[m verision 2.6.3
[33m05645b6[m enhanced analyser
[33m70c1d85[m fix mosaic mode and introduce unint 16
[33m0725e5c[m added expose mosaic drizzle factor for disk space saving
[33m89b6e2b[m version 2.5.5 unaligned file management
[33me9070fb[m info message unaligned
[33m5a64a57[m  drizzle fix fix infinite loop2
[33m780cfda[m  drizzle fix fix infinite loop
[33m1ee7fa0[m added support unint16 uint32
[33mb285964[m Merge branch 'main' of https://github.com/tinystork/zeseestarstacker into graphic-Beta
[33md6332a9[m Version 2.2.0 added astap support and astrometry local fix deletion of unaligned files
[33m2a1638e[m added ASTAP suport AND astrrometry support draft
[33m12e7545[m version 2.1.0 prise en charge ASTAP loacle
[33mc002a85[m Merge branch 'graphic-beta'
[33mfbf760d[m Nettoyage : suppression fichiers locaux et ajout du .gitignore ajout traduction mosaique
[33m62f37ac[m Merge branch 'graphic-Beta'
[33m7cd9b48[m correction modes drizzle
[33m163cc69[m Merge branch 'graphic-Beta'
[33m8f075b6[m update fall back 2.0.0
[33m0179959[m update json
[33m3e2dfa6[m update json
[33mc6eba02[m remove unused json
[33m58b2bc4[m Merge branch 'main' of https://github.com/tinystork/zeseestarstacker
[33m3d33edf[m version 2.0.0 added
[33m747c417[m clean up
[33m9124ae0[m Update seestar_settings.json
[33m1f952fc[m Update seestar_settings.json
[33m209e7c3[m Résolution du conflit de fusion dans mosaic processor
[33mb4775e8[m  version 2.0 mosaic, drizzle, standard
[33m542aaa9[m Modifications temporaires avant mise à jour de previous-version
[33m55d737c[m half local drizzle
[33m8b7c4a1[m[33m ([m[1;31morigin/previous-version[m[33m, [m[1;32mprevious-version[m[33m)[m restore mosaic standard and improves standard stack
[33m4ddcd58[m catchup
[33m04c9384[m drizzle incremntal fix
[33md5d322d[m restored standard stack drizzle final and mosaic
[33mb06f99d[m catchup
[33m8816113[m catchup
[33m9fb7063[m resolution conflit locale
[33m25dc05a[m drizzle fix mosaic fix
[33m75b7284[m Merge pull request #7 from tinystork/graphic-Beta
[33m4a6b8b2[m Merge branch 'main' into graphic-Beta
[33mbaec4b2[m fixed mosaic and drizzle
[33m7661c02[m Update requirements.txt
[33m856df75[m Update seestar_settings.json
[33m409a33e[m Update requirements.txt
[33m231cd62[m Update requirements.txt
[33md3e00bc[m Update requirements.txt
[33m3819217[m Update requirements.txt
[33m5a4abb0[m Update requirements.txt
[33m634a81d[m Update requirements.txt
[33mf7c8170[m Merge branch 'graphic-Beta'
[33m1e2f0c1[m  update readme
[33m5bb1e00[m Update readme.txt
[33mc6de35a[m pycache removal
[33m60d7ba2[m version stable color and stacking fixed with expert mode
[33m4ea208e[m added feather mode for weirdo artefacts between various batches
[33m5cd0a24[m Photo utils background normalisation added + expert tab
[33m17701e8[m Merge branch 'graphic-beta' fixed : green noise edges logic and mosaic modes
[33me54ad0e[m fixed green noise and mosaic mode
[33md28f04b[m Update README.md
[33ma23fc9c[m Update README.md
[33ma510275[m fix drizzle and add scnr treatment
[33m4cd15de[m Merge branch 'graphcBeta' Fixes the weird colors
[33mefd49f6[m weirdo colors fixed edge fixed
[33mf440d28[m weirdo colors fixed edge fixed
[33m497032c[m version mosaique par astrometry.net
[33m2a57823[m Mosaïc mode
[33m4536dad[m added edge color-enhancemement
[33m490026d[m Merge branch 'graphic-Beta' LATEST VERSION INTRODUCING AUTO ANALYSIS AND STACKING USE DETSAT WITH CAUTION
[33m82b6307[m added auto analysis and start stacking
[33m2a6257b[m Merge branch 'graphic-Beta' of https://github.com/tinystork/zeseestarstacker into graphic-Beta keep track
[33m9d3ac28[m json no big deal
[33m1a3b195[m Merge pull request #6 from tinystork/graphic-Beta
[33m93b6acd[m Merge branch 'main' into graphic-Beta
[33m3f8391d[m File analyser statdet and cleaner added
[33m8332223[m Merge graphic-Beta into main, accepting deletion of LICENSE file
[33macae135[m VERSION COMPLETE DRIZZLE STABLE
[33m9dee2de[m VERSION INTEGRANT INCREMENTAL DRIZZLE ET STANDARD DRIZZLE
[33mc60d27a[m VERSION INTEGRANT INREMENTAL DRIZZLE
[33m39b8563[m removed pycache
[33mf76ad63[m Drizzle standard and incremental added, fixed weird edges
[33m3322ca4[m drizzle not yet working
[33ma3ccd6a[m sauvegarde
[33m42c950f[m cosmestic
[33m93a62ef[m Merge branch 'main' of https://github.com/tinystork/zeseestarstacker
[33m0acfd6d[m update erro log & memory managment
[33madb019b[m Update README.md
[33m0461073[m added control button for aditionnal folder
[33m606113f[m readme updated with acknoledgments
[33m3fddc1c[m rm unused files
[33maa9b19d[m fixed cuda requirements, verbose anoying messages in terminal auto variance threshold
[33m4d9d6be[m Update README.md
[33m032fa0e[m Update README.md
[33me66ac61[m Update README.md
[33ma715bc2[m Update README.md
[33m5725480[m fixed cuda and warningsandheaderwarning
[33ma4b6909[m fixed cuda and warning for cuda fixxed requirements
[33mcb513c7[m remise à zero main
[33m7a56b76[m Merge branch 'main' of https://github.com/tinystork/zeseestarstacker
[33mc5c37cb[m remise à zero main
[33mb7f705d[m Delete seestar/gui/__pycache__ directory
[33m64f3e92[m Delete seestar/core/__pycache__ directory
[33m44dee11[m Delete seestar/localization/__pycache__ directory
[33me017582[m Delete seestar/queuep/__pycache__ directory
[33ma512049[m Delete seestar/tools/__pycache__ directory
[33md89dee9[m Delete seestar/__pycache__ directory
[33m0683fae[m Merge branch 'main' of https://github.com/tinystork/zeseestarstacker
[33mb0344a2[m rajout icones perdues
[33m81aae45[m update read me
[33mb2f5cda[m trial dataset
[33m213866b[m deplacement licence
[33m2f6a9a2[m deplacement readme
[33mb1a5e67[m restore missing images
[33m4afde89[m fichiers inutiles
[33m8a85b0a[m Update README.md
[33m5c35ebf[m Update README.md
[33m162bb5f[m updated read me
[33m7ec2fe1[m py cache removal
[33m3ba9c9d[m version 1.0 including snr weighting
[33mc6e6b41[m improved graphic improved logic
[33m2694a41[m Add files via upload
[33m2f15d2d[m Update README.md
[33m75d9e58[m Create README.md
[33me7041bb[m Initial update
[33m91771df[m Update README.md
[33m2767d4a[m Update README.md
[33me4afe3c[m first stable version
[33meed69c0[m Initial commit
