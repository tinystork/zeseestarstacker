# Seestar Stacker

📝 **Description**

*   **Français :**
    Seestar Stacker est un outil graphique (GUI) conçu pour aligner et empiler des images astronomiques FITS, typiquement issues du télescope Seestar S50. Il vise à améliorer le rapport signal/bruit par empilement par lots, tout en offrant une prévisualisation interactive et des options de correction.

*   **English:**
    Seestar Stacker is a Graphical User Interface (GUI) tool designed to align and stack astronomical FITS images, typically from the Seestar S50 telescope. It aims to improve the signal-to-noise ratio through batch stacking, while providing interactive preview and correction options.

---

✨ **Fonctionnalités / Features**

*   **Français :**
    *   Interface graphique conviviale (basée sur Tkinter).
    *   Alignement d'images FITS basé sur `astroalign`.
    *   Empilement par lots (Batch Stacking) utilisant différentes méthodes : Moyenne, Médiane, Kappa-Sigma Clipping, Winsorized Sigma Clipping.
    *   Prévisualisation interactive avec zoom/pan persistant.
    *   Ajustements de l'aperçu : Balance des blancs (manuelle/auto), Étirement d'histogramme (Linéaire, Asinh, Log; manuel/auto), Gamma, Luminosité, Contraste, Saturation.
    *   Histogramme interactif pour ajuster les points noir/blanc.
    *   Correction optionnelle des pixels chauds.
    *   Débayerisation automatique (basée sur motif GRBG par défaut).
    *   Traitement asynchrone en arrière-plan avec file d'attente (`QueueManager`).
    *   Possibilité d'ajouter des dossiers d'images supplémentaires pendant le traitement.
    *   Affichage de la progression (fichiers, lots, ETA).
    *   Localisation (Français / Anglais).
    *   Sauvegarde des paramètres utilisateur.
    *   Nettoyage optionnel des fichiers temporaires.

*   **English:**
    *   User-friendly Graphical User Interface (based on Tkinter).
    *   FITS image alignment based on `astroalign`.
    *   Batch Stacking using various methods: Mean, Median, Kappa-Sigma Clipping, Winsorized Sigma Clipping.
    *   Interactive preview with persistent zoom/pan.
    *   Preview adjustments: White Balance (manual/auto), Histogram Stretch (Linear, Asinh, Log; manual/auto), Gamma, Brightness, Contrast, Saturation.
    *   Interactive histogram for adjusting black/white points.
    *   Optional hot pixel correction.
    *   Automatic debayering (default GRBG pattern).
    *   Asynchronous background processing with a queue (`QueueManager`).
    *   Ability to add additional image folders during processing.
    *   Progress display (files, batches, ETA).
    *   Localization (French / English).
    *   User settings persistence.
    *   Optional cleanup of temporary files.

---

🛠️ **Installation**

1.  **Clonez ce dépôt / Clone this repository:**
    ```bash
    git clone https://github.com/tinystork/zeseestarstacker.git
    ```
    *(Note: Assurez-vous que l'URL du dépôt est correcte / Ensure the repository URL is correct)*

2.  **Accédez au répertoire du projet / Navigate to the project directory:**
    ```bash
    cd zeseestarstacker
    ```

3.  **Installez les dépendances requises / Install the required dependencies:**
    *(Il est recommandé d'utiliser un environnement virtuel / Using a virtual environment is recommended)*
    ```bash
    pip install -r requirements.txt
    ```
    *(Dépendances principales / Main dependencies: `numpy`, `opencv-python`, `astropy`, `astroalign`, `matplotlib`, `Pillow`. `psutil` est optionnel mais recommandé pour l'estimation auto de la taille des lots / `psutil` is optional but recommended for auto batch size estimation.)*

---

🚀 **Utilisation / Usage**

1.  Assurez-vous d'être dans le répertoire parent du dossier `seestar` (par exemple, dans le dossier `zeseestarstacker` cloné).
    *Make sure you are in the parent directory of the `seestar` folder (e.g., inside the cloned `zeseestarstacker` directory).*

2.  Exécutez le script principal / Run the main script:
    ```bash
    python seestar/main.py
    ```
    *(Alternativement, si vous êtes dans le dossier `seestar`: `python main.py`)*
    *(Alternatively, if inside the `seestar` directory: `python main.py`)*

3.  Utilisez l'interface graphique pour :
    *   Sélectionner le dossier d'entrée contenant vos fichiers FITS.
    *   Sélectionner un dossier de sortie pour les résultats.
    *   Choisir une image de référence (optionnel, sinon auto-sélection).
    *   Ajuster les options d'empilement (méthode, kappa, taille de lot, correction pixels chauds).
    *   Ajuster les paramètres de l'aperçu (balance des blancs, étirement, etc.).
    *   Cliquer sur "Démarrer" pour lancer le traitement.
    *   Ajouter des dossiers supplémentaires pendant le traitement si nécessaire via le bouton "Ajouter Dossier".

    *Use the GUI to:*
    *   *Select the input folder containing your FITS files.*
    *   *Select an output folder for the results.*
    *   *Choose a reference image (optional, otherwise auto-selected).*
    *   *Adjust stacking options (method, kappa, batch size, hot pixel correction).*
    *   *Adjust preview settings (white balance, stretch, etc.).*
    *   *Click "Start" to begin processing.*
    *   *Add additional folders during processing if needed using the "Add Folder" button.*

---

📂 **Structure du projet / Project Structure**

*   `seestar/` : Package principal de l'application / Main application package.
    *   `core/` : Logique de base (alignement, traitement image, pixels chauds, utilitaires) / Core logic (alignment, image processing, hot pixels, utilities).
    *   `gui/` : Interface graphique (fenêtre principale, gestionnaires UI, aperçu, histogramme, etc.) / GUI components (main window, UI managers, preview, histogram, etc.).
    *   `localization/` : Fichiers de traduction (en, fr) et classe de gestion / Translation files (en, fr) and management class.
    *   `queuep/` : Gestion du traitement asynchrone par lots via file d'attente / Asynchronous batch processing queue management.
    *   `tools/` : Outils auxiliaires (étirements, correction couleur) / Auxiliary tools (stretching, color correction).
    *   `main.py` : Script principal pour lancer l'interface graphique / Main script to launch the GUI.
*   `README.md` : Ce fichier / This file.
*   `requirements.txt` : Liste des dépendances Python / List of Python dependencies.
*   `seestar_settings.json` : Fichier de sauvegarde des paramètres utilisateur / User settings save file.

---

📜 **Licence / License**

Ce projet est sous licence GPL-3.0 / This project is licensed under the GPL-3.0 license.