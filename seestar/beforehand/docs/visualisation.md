# Visualisation des résultats / Results Visualization

Cette fenêtre permet d'explorer les données analysées grâce à plusieurs onglets interactifs. Les intitulés proviennent du fichier `zone.py` et sont disponibles en français et en anglais.

## Onglets

- **Distribution SNR** (`visu_tab_snr_dist` / `SNR Distribution`)
  - Histogramme du rapport signal/bruit. Utilisez le curseur pour sélectionner une plage et activer le bouton *Appliquer Rejet SNR*.
- **Distribution Starcount** (`starcount_distribution_tab` / `Starcount Distribution`)
  - Nombre d'étoiles détectées par image. Permet de filtrer via un curseur et d'appliquer un rejet Starcount.
- **Distribution FWHM** (`visu_tab_fwhm_dist` / `FWHM Distribution`)
  - Répartition de la taille apparente des étoiles. Un filtre FWHM peut être appliqué à l'aide du bouton dédié.
- **Distribution Excentricité** (`visu_tab_ecc_dist` / `Eccentricity Distribution`)
  - Histogramme de l'excentricité des étoiles. Un filtre d'excentricité est disponible.
- **FWHM vs e**
  - Nuage de points corrélant FWHM et excentricité.
- **Comparaison SNR** (`visu_tab_snr_comp` / `SNR Comparison`)
  - Barres horizontales montrant les meilleures et pires images selon le SNR.
- **Traînées Détectées** (`visu_tab_sat_trails` / `Detected Trails`)
  - Camembert de la proportion d'images avec ou sans traînée satellite.
- **Données Détaillées** (`visu_tab_raw_data` / `Detailed Data`)
  - Tableau récapitulatif de toutes les mesures. Les colonnes portent les noms définis dans `zone.py` (ex. `visu_data_col_file`, `visu_data_col_snr`).
- **Recommandations Stacking** (`visu_tab_recom` / `Stacking Recommendations`)
  - Liste des images recommandées pour l'empilement et boutons d'export.

## Boutons principaux

- **Appliquer Rejet SNR** (`visual_apply_snr_button` / `Apply SNR Rejection`)
- **Appliquer Rejet Starcount** (`apply_starcount_rejection` / `Apply Starcount Rejection`)
- **Filtre FWHM** (`filter_fwhm` / `FWHM Filter`)
- **Filtre Excentricité** (`filter_ecc` / `Eccentricity Filter`)
- **Appliquer Recommandations** (`visual_apply_reco_button` / `Apply Recommendations`)
- **Fermer** (`Fermer` / `Close`)
- **Exporter Liste Recommandée (.txt)** (`export_button`)
- **Exporter Toutes Conservées** (`Exporter Toutes Conservées` / `Export All Kept`)

Chaque bouton utilise les libellés définis dans `zone.py` pour s'afficher dans la langue choisie.
