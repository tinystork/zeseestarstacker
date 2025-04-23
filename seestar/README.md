# Seestar Stacker
**(English)** User-friendly Astronomical Image Stacker for Seestar S50
**(Français)** Empileur d'Images Astronomiques Convivial pour Seestar S50

**(English)**
Seestar Stacker is a graphical application designed to align and stack astronomical images captured with the Seestar S50 smart telescope (and potentially other FITS sources). Its primary goal is to improve the signal-to-noise ratio (SNR) of your astrophotography observations by combining multiple light frames.

**(Français)**
Seestar Stacker est une application graphique conçue pour aligner et empiler des images astronomiques capturées avec le télescope intelligent Seestar S50 (et potentiellement d'autres sources FITS). Son objectif principal est d'améliorer le rapport signal/bruit (SNR) de vos observations astrophotographiques en combinant plusieurs images brutes (light frames).

## Key Features / Fonctionnalités Clés

**(English)**

*   **FITS Loading & Validation:** Loads `.fit` and `.fits` files, performs basic validation, and handles common FITS format variations (e.g., channel order).
*   **Debayering:** Converts raw Bayer sensor data (RGGB, GRBG, etc.) into color images using OpenCV.
*   **Hot Pixel Correction:** Detects and corrects hot pixels based on local statistics (median/std dev).
*   **Image Alignment:** Automatically finds a suitable reference frame (or uses a user-provided one) and aligns subsequent images using star patterns via `astroalign`.
*   **Stacking Methods:** Offers several stacking algorithms: Mean, Median, Kappa-Sigma Clipping, Winsorized Sigma Clipping.
*   **Quality Weighting (Optional but Recommended):** Analyzes each aligned frame based on Signal-to-Noise Ratio (SNR) and Star Count/Sharpness, assigns weights, and allows tuning via exponents and minimum weight.
*   **Asynchronous Processing:** Performs alignment and stacking in a background thread, keeping the GUI responsive.
*   **Batch Processing:** Processes images in memory-efficient batches.
*   **Graphical User Interface (Tkinter):**
    *   Intuitive tabbed interface.
    *   Live Preview with interactive zoom/pan and adjustments (White Balance, Stretch, Gamma, BCS).
    *   Interactive Histogram with draggable Black/White point markers.
    *   Progress Tracking (bar, status, ETA, elapsed time).
    *   Multi-language Support (English, French).
    *   Customizable: Supports custom application icon and preview background image.
*   **Configuration:** Saves and loads user settings (`seestar_settings.json`).
*   **Workflow Tools:** Add folders during processing, Copy Log button, Open Output Folder button, optional temporary file cleanup.

**(Français)**

*   **Chargement & Validation FITS :** Charge les fichiers `.fit` et `.fits`, effectue une validation de base et gère les variations courantes du format FITS (ex: ordre des canaux).
*   **Débayerisation :** Convertit les données brutes du capteur Bayer (RGGB, GRBG, etc.) en images couleur via OpenCV.
*   **Correction Pixels Chauds :** Détecte et corrige les pixels chauds en se basant sur les statistiques locales (médiane/écart-type).
*   **Alignement d'Images :** Trouve automatiquement une image de référence appropriée (ou utilise celle fournie par l'utilisateur) et aligne les images suivantes en utilisant les motifs d'étoiles via `astroalign`.
*   **Méthodes d'Empilement :** Offre plusieurs algorithmes : Moyenne, Médiane, Kappa-Sigma Clipping, Winsorized Sigma Clipping.
*   **Pondération par Qualité (Optionnel mais recommandé) :** Analyse chaque image alignée selon le Rapport Signal/Bruit (SNR) et le Nombre/Netteté des Étoiles, assigne des poids et permet l'ajustement via des exposants et un poids minimum.
*   **Traitement Asynchrone :** Effectue l'alignement et l'empilement dans un thread d'arrière-plan, gardant l'interface graphique réactive.
*   **Traitement par Lots :** Traite les images par lots efficaces en mémoire.
*   **Interface Graphique (Tkinter) :**
    *   Interface intuitive à onglets.
    *   Aperçu en direct avec zoom/pan interactif et ajustements (Balance des Blancs, Étirement, Gamma, Luminosité/Contraste/Saturation).
    *   Histogramme interactif avec marqueurs de point Noir/Blanc déplaçables.
    *   Suivi de Progression (barre, statut, ETA, temps écoulé).
    *   Support Multilingue (Anglais, Français).
    *   Personnalisable : Supporte une icône d'application et une image de fond d'aperçu personnalisées.
*   **Configuration :** Sauvegarde et charge les paramètres utilisateur (`seestar_settings.json`).
*   **Outils de Workflow :** Ajout de dossiers pendant le traitement, bouton Copier Log, bouton Ouvrir Dossier Sortie, nettoyage optionnel des fichiers temporaires.

---

## Requirements / Prérequis

**(English)**

*   **Python:** 3.8 or higher recommended.
*   **Required Packages:** Listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install numpy opencv-python astropy astroalign tqdm psutil matplotlib Pillow scikit-image
    ```
    *   `psutil` is optional but highly recommended for automatic batch size estimation.

**(Français)**

*   **Python :** 3.8 ou supérieur recommandé.
*   **Packages Requis :** Listés dans `requirements.txt`. Installez-les avec pip :
    ```bash
    pip install numpy opencv-python astropy astroalign tqdm psutil matplotlib Pillow scikit-image
    ```
    *   `psutil` est optionnel mais fortement recommandé pour l'estimation automatique de la taille des lots.

---

## Installation / Setup

**(English)**

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/tinystork/zeseestarstacker.git
    cd zeseestarstacker
    ```
2.  **(Recommended)** Create and activate a virtual environment:
    ```bash
    # Windows
    python -m venv .venv
    .\.venv\Scripts\activate

    # macOS / Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **(Optional - Customization):**
    *   Place your custom window icon (`.png`, e.g., 256x256) and update the `icon_path` in `seestar/gui/main_window.py`.
    *   Place your custom background image (`.png` or `.jpg`) and update the `bg_image_path` in `seestar/gui/preview.py`.

**(Français)**

1.  **Cloner le Dépôt :**
    ```bash
    git clone https://github.com/tinystork/zeseestarstacker.git
    cd zeseestarstacker
    ```
2.  **(Recommandé)** Créer et activer un environnement virtuel :
    ```bash
    # Windows
    python -m venv .venv
    .\.venv\Scripts\activate

    # macOS / Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  **Installer les Dépendances :**
    ```bash
    pip install -r requirements.txt
    ```
4.  **(Optionnel - Personnalisation) :**
    *   Placez votre icône de fenêtre personnalisée (`.png`, ex: 256x256) et mettez à jour `icon_path` dans `seestar/gui/main_window.py`.
    *   Placez votre image de fond personnalisée (`.png` ou `.jpg`) et mettez à jour `bg_image_path` dans `seestar/gui/preview.py`.

---

## Usage / Utilisation

**(English)**

1.  **Run the Application:** Navigate to the project's root directory and run `python seestar/main.py`.
2.  **Select Folders:** Choose Input, Output, and optional Reference folders/files.
3.  **Adjust Stacking Settings:** Select Method, Kappa, Batch Size, Hot Pixel options, Quality Weighting parameters, and Cleanup preference.
4.  **Adjust Preview (Optional):** Modify preview display settings (WB, Stretch, Gamma, etc.) using the Preview tab and Histogram. *This does not affect the final FITS stack.*
5.  **Start Processing:** Click the "Start" button.
6.  **Monitor Progress:** Watch the progress bar, log messages, timers, and live preview.
7.  **Add Folders (Optional):** Click "Add Folder" during processing to queue additional image sets.
8.  **Completion:** Review the summary dialog. Find the final `stack_final_....fit` and preview PNG in the output folder. Use "Copy Log" / "Open Output" buttons if needed.

**(Français)**

1.  **Lancer l'Application :** Naviguez vers le répertoire racine du projet et exécutez `python seestar/main.py`.
2.  **Sélectionner les Dossiers :** Choisissez les dossiers d'Entrée, de Sortie et optionnellement le fichier de Référence.
3.  **Ajuster les Paramètres d'Empilement :** Sélectionnez la Méthode, Kappa, Taille de Lot, options de Pixels Chauds, paramètres de Pondération par Qualité et préférence de Nettoyage.
4.  **Ajuster l'Aperçu (Optionnel) :** Modifiez les paramètres d'affichage de l'aperçu (BdB, Étirement, Gamma, etc.) via l'onglet Aperçu et l'Histogramme. *Ceci n'affecte pas le stack FITS final sauvegardé.*
5.  **Démarrer le Traitement :** Cliquez sur le bouton "Démarrer".
6.  **Suivre la Progression :** Observez la barre de progression, les messages du log, les chronomètres et l'aperçu en direct.
7.  **Ajouter des Dossiers (Optionnel) :** Cliquez sur "Ajouter Dossier" pendant le traitement pour mettre en file d'attente des jeux d'images additionnels.
8.  **Fin :** Consultez le dialogue de résumé. Trouvez l'image finale `stack_final_....fit` et l'aperçu PNG dans le dossier de sortie. Utilisez les boutons "Copier Log" / "Ouvrir Sortie" si besoin.

---

## Configuration (`seestar_settings.json`)

**(English)**
The application saves your settings (paths, processing parameters, preview adjustments, UI state) to `seestar_settings.json`. Deleting this file resets settings to default.

**(Français)**
L'application sauvegarde vos paramètres (chemins, paramètres de traitement, ajustements d'aperçu, état de l'interface) dans `seestar_settings.json`. Supprimer ce fichier réinitialise les paramètres par défaut.

---

## Understanding Quality Weighting / Comprendre la Pondération par Qualité

**(English)**
Quality weighting aims to improve the final stack by giving more importance to the "best" images. It analyzes frames based on SNR (Signal-to-Noise Ratio) and Star Count/Sharpness. Use the GUI controls (Enable, Metrics, Exponents, Min Weight) to tune its behavior. See `quality weighting explained.txt` for details.

**(Français)**
La pondération par qualité vise à améliorer le stack final en donnant plus d'importance aux "meilleures" images. Elle analyse les images selon le SNR (Rapport Signal/Bruit) et le Nombre/Netteté des Étoiles. Utilisez les contrôles de l'interface (Activer, Métriques, Exposants, Poids Min) pour ajuster son comportement. Voir `quality weighting explained.txt` pour les détails.

---

## Troubleshooting / FAQ / Dépannage

**(English)**

*   **Error: Dependencies Missing:** Run `pip install -r requirements.txt`.
*   **GUI Doesn't Appear / Display Error:** Ensure you are running in a graphical environment.
*   **Images Fail to Align:** Check logs. Causes: poor seeing, clouds, trailing, few stars, bad reference. Try auto-reference first. Inspect "unaligned" files (if cleanup disabled).
*   **Warning: Low Variance:** Normal for very dark/cloudy frames.
*   **App Starts with Invalid Paths:** Use "Browse" to select valid paths before starting. The app checks paths on "Start".
*   **Quality Weighting Issues:** Try lowering exponents (< 1.0) or disabling one metric (SNR/Stars). Ensure Min Weight is low (0.05-0.15).

**(Français)**

*   **Erreur : Dépendances Manquantes :** Exécutez `pip install -r requirements.txt`.
*   **L'Interface n'apparaît pas / Erreur d'Affichage :** Assurez-vous d'exécuter dans un environnement graphique.
*   **Échec d'Alignement des Images :** Vérifiez les logs. Causes : mauvaise météo, nuages, filé d'étoiles, peu d'étoiles, mauvaise référence. Essayez d'abord la référence auto. Inspectez les fichiers "unaligned" (si nettoyage désactivé).
*   **Avertissement : Faible Variance :** Normal pour images très sombres/nuageuses.
*   **L'App Démarre avec des Chemins Invalides :** Utilisez "Parcourir" pour sélectionner des chemins valides avant de démarrer. L'app vérifie les chemins au "Démarrage".
*   **Problèmes Pondération Qualité :** Essayez de baisser les exposants (< 1.0) ou désactivez une métrique (SNR/Étoiles). Assurez-vous que Poids Min est bas (0.05-0.15).

---

## License / Licence

**(English)**
This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

**(Français)**
Ce projet est sous licence GNU General Public License v3.0. Voir le fichier [LICENSE](LICENSE) pour les détails.

---

## Author / Auteur

*   **Tinystork**