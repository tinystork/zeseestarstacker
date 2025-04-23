# Seestar Stacker the little stacker for a lot of light :-)
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
## Optional GPU Acceleration (CUDA)

This application can optionally leverage an NVIDIA GPU using CUDA to accelerate certain processing steps, potentially leading to significant speed improvements, especially for stacking large numbers of images.

**However, GPU acceleration is strictly optional.** The application will run correctly using your computer's main processor (CPU) by default, without requiring any special hardware or setup.

### Requirements for GPU Acceleration

To enable GPU acceleration, you need:

1.  **Hardware:** An NVIDIA graphics card that supports CUDA.
2.  **Drivers:** Up-to-date NVIDIA drivers for your graphics card.
3.  **CUDA Toolkit:** The NVIDIA CUDA Toolkit installed on your system. You can check if it's installed and find its version by opening a terminal or command prompt and running `nvcc --version`.

### Python Dependencies for GPU Acceleration

If you meet the requirements above, you need specific Python packages:

1.  **OpenCV Contrib:** The `requirements.txt` file includes `opencv-contrib-python`. This version *might* use CUDA for some operations (like denoising) if your system is correctly configured *before* you install it via pip.
2.  **CuPy:** For significantly faster image stacking, you need to install CuPy *manually* after installing the base requirements.
    *   Check your CUDA Toolkit version (`nvcc --version`).
    *   Install the matching CuPy package. See the detailed comments in the `requirements.txt` file for the exact `pip install cupy-cudaXXX` command corresponding to your CUDA version (e.g., `pip install cupy-cuda12x` for CUDA 12.x).

### Important Note on Terminal Messages (GPU Checks)

When you run the application, it performs checks to see if CUDA-enabled OpenCV and CuPy are available.

*   You might see messages in your terminal like:
    *   `DEBUG: CUDA device(s) detected by OpenCV.` (or `No CUDA devices detected...`)
    *   `DEBUG: CuPy library not found.`
    *   `DEBUG: CuPy detected CUDA Device X: ...` (or `No CUDA device is available/detected by CuPy.`)
    *   `Warning: CUDA ... failed: ... Falling back to CPU.`
    *   Errors related to missing `.dll` files (like `nvrtc64_XXX.dll`) if CuPy is installed but doesn't match your CUDA Toolkit.

*   **If you do NOT have an NVIDIA GPU or have not installed the CUDA Toolkit and the correct CuPy package, seeing these messages is NORMAL and EXPECTED.**
*   They simply indicate that the optional GPU acceleration could not be activated.
*   **The application is designed to automatically and safely fall back to using the CPU in these cases.** It will continue to function correctly.

---

## Accélération GPU Optionnelle (CUDA) - French Version

Cette application peut optionnellement utiliser une carte graphique NVIDIA via CUDA pour accélérer certaines étapes de traitement, ce qui peut améliorer significativement la vitesse, notamment pour l'empilement d'un grand nombre d'images.

**Cependant, l'accélération GPU est strictement optionnelle.** L'application fonctionnera correctement en utilisant le processeur principal de votre ordinateur (CPU) par défaut, sans nécessiter de matériel ou de configuration spécifique.

### Prérequis pour l'Accélération GPU

Pour activer l'accélération GPU, vous avez besoin de :

1.  **Matériel :** Une carte graphique NVIDIA supportant CUDA.
2.  **Pilotes :** Des pilotes NVIDIA à jour pour votre carte graphique.
3.  **CUDA Toolkit :** Le NVIDIA CUDA Toolkit installé sur votre système. Vous pouvez vérifier s'il est installé et connaître sa version en ouvrant un terminal ou une invite de commande et en exécutant `nvcc --version`.

### Dépendances Python pour l'Accélération GPU

Si vous remplissez les conditions ci-dessus, vous avez besoin de paquets Python spécifiques :

1.  **OpenCV Contrib :** Le fichier `requirements.txt` inclut `opencv-contrib-python`. Cette version *peut potentiellement* utiliser CUDA pour certaines opérations (comme le débruitage) si votre système est correctement configuré *avant* son installation via pip.
2.  **CuPy :** Pour un empilement d'images significativement plus rapide, vous devez installer CuPy *manuellement* après avoir installé les dépendances de base.
    *   Vérifiez la version de votre CUDA Toolkit (`nvcc --version`).
    *   Installez le paquet CuPy correspondant. Consultez les commentaires détaillés dans le fichier `requirements.txt` pour la commande exacte `pip install cupy-cudaXXX` adaptée à votre version CUDA (ex: `pip install cupy-cuda12x` pour CUDA 12.x).

### Note Importante Concernant les Messages du Terminal (Vérifications GPU)

Lorsque vous lancez l'application, elle effectue des vérifications pour voir si OpenCV avec CUDA et CuPy sont disponibles.

*   Vous pourriez voir des messages dans votre terminal tels que :
    *   `DEBUG: CUDA device(s) detected by OpenCV.` (ou `No CUDA devices detected...`)
    *   `DEBUG: CuPy library not found.`
    *   `DEBUG: CuPy detected CUDA Device X: ...` (ou `No CUDA device is available/detected by CuPy.`)
    *   `Warning: CUDA ... failed: ... Falling back to CPU.`
    *   Des erreurs liées à des fichiers `.dll` manquants (comme `nvrtc64_XXX.dll`) si CuPy est installé mais ne correspond pas à votre CUDA Toolkit.

*   **Si vous N'AVEZ PAS de GPU NVIDIA ou si vous n'avez pas installé le CUDA Toolkit et le paquet CuPy correct, voir ces messages est NORMAL et ATTENDU.**
*   Ils indiquent simplement que l'accélération GPU optionnelle n'a pas pu être activée.
*   **L'application est conçue pour utiliser automatiquement et sans danger le CPU dans ces cas.** Elle continuera de fonctionner correctement.
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
