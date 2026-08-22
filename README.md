# ZeSeestaStacker the little stacker for a lot of light
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
*   **Winsorized Sigma Clip:** Rejected pixels are automatically replaced with the winsorized limits.
*   **Quality Weighting (Optional but Recommended):** Analyzes each aligned frame based on Signal-to-Noise Ratio (SNR) and Star Count/Sharpness, assigns weights, and allows tuning via exponents and minimum weight.
*   **Asynchronous Processing:** Performs alignment and stacking in a background thread, keeping the GUI responsive.
*   **Thread-Safe Event Queue:** Worker threads post GUI updates to an internal queue polled on the Tk thread.
*   **Batch Processing:** Processes images in memory-efficient batches.
*   **Graphical User Interface (Tkinter):**
    *   Intuitive tabbed interface.
    *   Live Preview with interactive zoom/pan and adjustments (White Balance, Stretch, Gamma, BCS).
    *   Interactive Histogram with draggable Black/White point markers.
    *   Progress Tracking (bar, status, ETA, elapsed time).
    *   Multi-language Support (English, French).
    *   Customizable: Supports custom application icon and preview background image.
*   **Configuration:** Saves and loads user settings (`seestar_settings.json`).
*   **Output Format:** Save FITS stacks as float32 or uint16. Enable *Preserve Linear Output* to skip percentile scaling.
*   **Workflow Tools:** Add folders during processing, Copy Log button, Open Output Folder button, optional temporary file cleanup.
*   **Smart Quality Control:**
    *   SNR-based weighting
    *   Star count/sharpness analysis
*   **Memory Optimization:** Batch processing with auto RAM management
*   **Batch Size 0 Mode:** If a `stack_plan.csv` file is present in the input
    folder, its `batch_id` column controls grouping when **Batch Size** is set
    to `0`. Otherwise the backend automatically estimates a suitable size.
*   **Threaded Boring Stack Mode:** Set **Batch Size** to `1` (or enable the
    *Threaded Boring Stack* checkbox) to run `boring_stack.py` in a dedicated
    worker thread for minimal memory usage.

The Expert tab's **Output FITS Format** panel features a checkbox labeled
**Preserve Linear Output**. Enabling it saves the stacked FITS directly from the
linear data, skipping the usual percentile scaling step so pixel values are
written linearly.

**(Français)**

*   **Chargement & Validation FITS :** Charge les fichiers `.fit` et `.fits`, effectue une validation de base et gère les variations courantes du format FITS (ex: ordre des canaux).
*   **Débayerisation :** Convertit les données brutes du capteur Bayer (RGGB, GRBG, etc.) en images couleur via OpenCV.
*   **Correction Pixels Chauds :** Détecte et corrige les pixels chauds en se basant sur les statistiques locales (médiane/écart-type).
*   **Alignement d'Images :** Trouve automatiquement une image de référence appropriée (ou utilise celle fournie par l'utilisateur) et aligne les images suivantes en utilisant les motifs d'étoiles via `astroalign`.
*   **Méthodes d'Empilement :** Offre plusieurs algorithmes : Moyenne, Médiane, Kappa-Sigma Clipping, Winsorized Sigma Clipping.
*   **Winsorized Sigma Clip :** Les pixels rejetés sont automatiquement remplacés par les limites winsorisées.
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
*   **Format de Sortie :** Sauvegarde des FITS en float32 ou uint16. Activez *Préserver l'image linéaire* pour éviter la normalisation par percentiles.
*   **Outils de Workflow :** Ajout de dossiers pendant le traitement, bouton Copier Log, bouton Ouvrir Dossier Sortie, nettoyage optionnel des fichiers temporaires.
*   **Contrôle Qualité Intelligent :**
    *   Pondération par rapport signal/bruit (SNR)
    *   Analyse du nombre/netteté des étoiles
*   **Optimisation Mémoire :** Traitement par lots avec gestion automatique de la RAM

Dans l'onglet **Expert**, la section **Format de sortie FITS** propose une case
à cocher **Sortie linéaire préservée**. Une fois activée, la sauvegarde FITS se
fait sans normalisation par percentiles et les données sont écrites linéairement.

---

## Requirements / Prérequis

**(English)**

*   **Version:** ZeSeestarStacker 8.0.0 beta, codename **Phoenix consedit**.
*   **Operating systems:** Windows, Linux, or macOS.
*   **Python:** 3.10 or higher.
*   **Recommended GUI:** the PySide6 / Qt interface (included in a plain install).
*   **PyQt5:** not required for the main Qt application.
*   **GPU:** not required. ZeSeestarStacker runs normally on CPU. NVIDIA GPU acceleration is optional and can be configured separately later.

**(Français)**

*   **Version :** ZeSeestarStacker 8.0.0 beta, nom de code **Phoenix consedit**.
*   **Systèmes :** Windows, Linux ou macOS.
*   **Python :** 3.10 ou supérieur.
*   **Interface recommandée :** l'interface PySide6 / Qt (incluse dans l'installation standard).
*   **PyQt5 :** non requis pour l'application Qt principale.
*   **GPU :** non obligatoire. ZeSeestarStacker fonctionne normalement sur CPU. L'accélération NVIDIA GPU est optionnelle et peut être configurée séparément plus tard.

---

## Installation / Setup

**(English)**

Install the current 8.0 beta directly from GitHub. You do **not** need to clone the repository.

### Windows — PowerShell

```powershell
py -m venv zsss-env
.\zsss-env\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install "ZeSeestarStacker @ https://github.com/tinystork/zeseestarstacker/archive/refs/heads/beta.zip"

zeseestarstacker
```

Later, to launch ZeSeestarStacker again from the same folder:

```powershell
.\zsss-env\Scripts\Activate.ps1
zeseestarstacker
```

### Linux / macOS

```bash
python3 -m venv zsss-env
source zsss-env/bin/activate

python -m pip install --upgrade pip
python -m pip install "ZeSeestarStacker @ https://github.com/tinystork/zeseestarstacker/archive/refs/heads/beta.zip"

zeseestarstacker
```

Later, to launch ZeSeestarStacker again from the same folder:

```bash
source zsss-env/bin/activate
zeseestarstacker
```

**Qt entry point:** Qt / PySide6 is now the official interface for this beta. The `zeseestarstacker` command launches the Qt GUI with the real engine by default. For a safe simulated/dev run without the engine, use `python -m seestar.qt_main --backend simulated`.

**(Français)**

Installez directement la beta 8.0 depuis GitHub. Vous n'avez **pas** besoin de cloner le dépôt.

### Windows — PowerShell

```powershell
py -m venv zsss-env
.\zsss-env\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install "ZeSeestarStacker @ https://github.com/tinystork/zeseestarstacker/archive/refs/heads/beta.zip"

zeseestarstacker
```

Plus tard, pour relancer ZeSeestarStacker depuis le même dossier :

```powershell
.\zsss-env\Scripts\Activate.ps1
zeseestarstacker
```

### Linux / macOS

```bash
python3 -m venv zsss-env
source zsss-env/bin/activate

python -m pip install --upgrade pip
python -m pip install "ZeSeestarStacker @ https://github.com/tinystork/zeseestarstacker/archive/refs/heads/beta.zip"

zeseestarstacker
```

Plus tard, pour relancer ZeSeestarStacker depuis le même dossier :

```bash
source zsss-env/bin/activate
zeseestarstacker
```

**Entry point Qt :** Qt / PySide6 est désormais l'interface officielle de cette beta. La commande `zeseestarstacker` lance l'interface Qt avec le moteur réel par défaut. Pour une exécution simulée de développement sans moteur, utilisez `python -m seestar.qt_main --backend simulated`.


## Development Setup

**Install dependencies**

```bash
# install all runtime and test dependencies
pip install -r requirements.txt -r requirements-test.txt
```

### Test Dependencies

The unit tests rely on a few additional packages:

- `pytest`
- `numpy`
- `astropy`
- `reproject`
- `drizzle` *(required for the Drizzle-related tests)*
- `psutil` *(only required when testing the optional memory usage features)*

Most of these packages are provided in `requirements-test.txt`. The
`drizzle` package is listed in `requirements.txt` and must be installed
for the tests that exercise the Drizzle functionality.

**Run the tests**

```bash
SEESTAR_VERBOSE=1 pytest -q
```

The `SEESTAR_VERBOSE` variable is optional and simply enables more verbose logs.

## Running Tests

Install all dependencies (including the optional ones used in the tests) with:

```bash
pip install -r requirements.txt -r requirements-test.txt
```

Then execute the tests with `pytest`:

```bash
pytest -q
```


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

1.  **Run the Application:** activate your environment, then start the Qt GUI with `zeseestarstacker`.
    Set the environment variable `SEESTAR_VERBOSE=1` (or pass `-v` when running
    `run_zemosaic.py`) to enable verbose debug output.
2.  **Select Folders:** Choose Input, Output, and optional Reference folders/files.
3.  **Adjust Stacking Settings:** Select Method, Kappa, Batch Size, Hot Pixel options, Quality Weighting parameters, and Cleanup preference.
4.  **Adjust Preview (Optional):** Modify preview display settings (WB, Stretch, Gamma, etc.) using the Preview tab and Histogram. *This does not affect the final FITS stack.*
5.  **Start Processing:** Click the "Start" button.
6.  **Monitor Progress:** Watch the progress bar, log messages, timers, and live preview.
7.  **Add Folders (Optional):** Click "Add Folder" during processing to queue additional image sets.
8.  **Completion:** Review the summary dialog. Find the final `stack_final_....fit` and preview PNG in the output folder. Use "Copy Log" / "Open Output" buttons if needed.

**(Français)**

1.  **Lancer l'Application :** activez votre environnement, puis lancez l'interface Qt avec `zeseestarstacker`.
    Définissez la variable d'environnement `SEESTAR_VERBOSE=1` (ou utilisez
    l'option `-v` avec `run_zemosaic.py`) pour activer les messages de débogage détaillés.
2.  **Sélectionner les Dossiers :** Choisissez les dossiers d'Entrée, de Sortie et optionnellement le fichier de Référence.
3.  **Ajuster les Paramètres d'Empilement :** Sélectionnez la Méthode, Kappa, Taille de Lot, options de Pixels Chauds, paramètres de Pondération par Qualité et préférence de Nettoyage.
4.  **Ajuster l'Aperçu (Optionnel) :** Modifiez les paramètres d'affichage de l'aperçu (BdB, Étirement, Gamma, etc.) via l'onglet Aperçu et l'Histogramme. *Ceci n'affecte pas le stack FITS final sauvegardé.*
5.  **Démarrer le Traitement :** Cliquez sur le bouton "Démarrer".
6.  **Suivre la Progression :** Observez la barre de progression, les messages du log, les chronomètres et l'aperçu en direct.
7.  **Ajouter des Dossiers (Optionnel) :** Cliquez sur "Ajouter Dossier" pendant le traitement pour mettre en file d'attente des jeux d'images additionnels.
8.  **Fin :** Consultez le dialogue de résumé. Trouvez l'image finale `stack_final_....fit` et l'aperçu PNG dans le dossier de sortie. Utilisez les boutons "Copier Log" / "Ouvrir Sortie" si besoin.

### Command-Line Mosaic

You can also run the hierarchical mosaic workflow directly from a terminal:

```bash
python -m seestar.scripts.run_mosaic INPUT_DIR OUTPUT_DIR \
    --astap-path /path/to/astap \
    --astap-data-dir /path/to/catalogs
```

Solver options correspond to the `ZEMOSAIC_*` environment variables used by the GUI. These values can be configured from the **Mosaic Options** window.
When calling the worker from Python, gather them into a `solver_settings` dictionary:

```python
from zemosaic import zemosaic_worker

solver_settings = {
    "astap_path": "/path/to/astap",
    "astap_data_dir": "/path/to/catalogs",
    "astap_search_radius": 3.0,
    "astap_downsample": 2,
    "astap_sensitivity": 100,
    "use_radec_hints": False,
    "local_ansvr_path": "/path/to/ansvr.cfg",
    "api_key": "your_key",
    "local_solver_preference": "astap",
}

zemosaic_worker.run_hierarchical_mosaic(
    "INPUT_DIR", "OUTPUT_DIR", solver_settings, cluster_threshold=0.5, ...
)
```

**`solver_settings` keys**

- `astap_path`: path to the ASTAP executable
- `astap_data_dir`: directory containing ASTAP star catalogs
- `astap_search_radius`: search radius in degrees passed to ASTAP. When
  omitted the solver uses `ASTAP_DEFAULT_SEARCH_RADIUS` (3.0 degrees)
- `astap_downsample`: downsample factor used by ASTAP
- `astap_sensitivity`: detection sensitivity percentage for ASTAP
- `use_radec_hints`: include FITS RA/DEC as hints when solving with ASTAP *(defaults to false; enable only if your FITS headers contain reliable coordinates)*
- `local_ansvr_path`: path to a local `ansvr.cfg`
- `ansvr_host_port`: host and port for a running ansvr instance (default `127.0.0.1:8080`)
- `astrometry_solve_field_dir`: directory containing the `solve-field` executable
- `api_key`: astrometry.net API key
- `local_solver_preference`: preferred local solver (`astap` or `ansvr`)

- `reproject_between_batches`: when enabled the frames of each batch are stacked
  and reprojected onto the reference WCS solved from the initial reference
  image. Intermediate stacks are not solved again unless `solve_batches` is
  `true`.

- `freeze_reference_wcs`: when `true` the reference WCS determined from the
  first solved batch remains fixed for the whole run, preventing small drifts
  between batches when using inter-batch reprojection. This is automatically
  enabled when `reproject_between_batches` is used unless explicitly set to
  `false` before starting the process.
- `solve_batches`: disable solving of each stacked batch when set to `false`.
  The stored reference WCS header is applied instead.

`use_radec_hints` controls whether ASTAP receives the RA/DEC coordinates from
the FITS header. This option is **disabled by default** and should only be
enabled when those header values are trustworthy. When disabled the solver
performs a blind search centered only on the provided search radius.

### Inter-Batch Reprojection with Constant WCS

You can reproject each stacked batch without solving them all. Enable

`reproject_between_batches` and set `solve_batches` to `false` so only the
reference image is solved once at the start. `freeze_reference_wcs` will be
turned on automatically in this mode (unless you disable it explicitly) so the
initial WCS is reused for every batch. Subsequent batches are reprojected using
a new fixed-orientation grid so the image orientation stays constant and small
rotation drifts are eliminated.

### Alignment Options

The aligner normally keeps intermediate arrays in memory. For very large
images you can reduce memory usage by performing alignment on disk-backed
``.npy`` files. Pass ``align_on_disk=True`` when using ``SeestarQueuedStacker``
from Python or add ``--align-on-disk`` to ``boring_stack.py``. When more than
50 images are queued and this flag is omitted, ``boring_stack.py`` now
automatically enables disk-backed alignment to avoid running out of memory.

### Command-Line Stacking

Use `seestar/gui/boring_stack.py` for headless single-batch processing. Set
`--chunk-size` to periodically flush intermediate stacks. When the GUI launches
`boring_stack.py` with **Batch Size** set to `1`, this option is automatically
added based on available system RAM:

```bash
python seestar/gui/boring_stack.py --csv stack_plan.csv --out OUT_DIR \
    --batch-size 1 --chunk-size 50
```


When `--chunk-size` is used with `--batch-size 1`, results are combined in
chunks of N images. Any `stack_plan.csv` present is honoured so files are
grouped by `batch_id` before chunking.

When using `--batch-size 1` together with `--align-on-disk`, aligned frames are
written to an `aligned_tmp` directory and reused when the final stack is
assembled. These temporary files are cleaned up automatically once processing
finishes.

### Threaded Boring Stack from the GUI

Set **Batch Size** to `1` or tick the *Threaded Boring Stack* checkbox in the
Stacking tab. The GUI launches `boring_stack.py` in a background thread using
your `stack_plan.csv` and displays progress as it runs. A chunk size is
automatically calculated and passed to the script so memory usage stays
bounded. When calling `boring_stack.py` directly, specifying `--chunk-size`

honours any grouping defined in `stack_plan.csv`.



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
Experimental weighting methods based on noise variance or FWHM are also available.

**(Français)**
La pondération par qualité vise à améliorer le stack final en donnant plus d'importance aux "meilleures" images. Elle analyse les images selon le SNR (Rapport Signal/Bruit) et le Nombre/Netteté des Étoiles. Utilisez les contrôles de l'interface (Activer, Métriques, Exposants, Poids Min) pour ajuster son comportement. Voir `quality weighting explained.txt` pour les détails.

---

## Troubleshooting / FAQ / Dépannage

**(English)**

*   **Error: Dependencies Missing:** Re-run the installation command for your system (a plain install already includes the Qt/PySide6 interface).
*   **GUI Doesn't Appear / Display Error:** Ensure you are running in a graphical environment.
*   **Images Fail to Align:** Check logs. Causes: poor seeing, clouds, trailing, few stars, bad reference. Try auto-reference first. Inspect "unaligned" files (if cleanup disabled).
*   **Warning: Low Variance:** Normal for very dark/cloudy frames.
*   **App Starts with Invalid Paths:** Use "Browse" to select valid paths before starting. The app checks paths on "Start".
*   **Quality Weighting Issues:** Try lowering exponents (< 1.0) or disabling one metric (SNR/Stars). Ensure Min Weight is low (0.05-0.15).
*   **Zero Coverage Warning:** If logs show `Cumulative weight map sums to zero`, no pixels were accumulated. Verify all batch images share the expected shape and that reprojection is configured correctly.

**(Français)**

*   **Erreur : dépendances manquantes :** relancez la commande d’installation correspondant à votre système (l'installation standard inclut déjà l'interface Qt/PySide6).
*   **L'Interface n'apparaît pas / Erreur d'Affichage :** Assurez-vous d'exécuter dans un environnement graphique.
*   **Échec d'Alignement des Images :** Vérifiez les logs. Causes : mauvaise météo, nuages, filé d'étoiles, peu d'étoiles, mauvaise référence. Essayez d'abord la référence auto. Inspectez les fichiers "unaligned" (si nettoyage désactivé).
*   **Avertissement : Faible Variance :** Normal pour images très sombres/nuageuses.
*   **L'App Démarre avec des Chemins Invalides :** Utilisez "Parcourir" pour sélectionner des chemins valides avant de démarrer. L'app vérifie les chemins au "Démarrage".
*   **Problèmes Pondération Qualité :** Essayez de baisser les exposants (< 1.0) ou désactivez une métrique (SNR/Étoiles). Assurez-vous que Poids Min est bas (0.05-0.15).
*   **Avertissement Couverture Zéro :** Si les logs affichent `Carte de poids cumulée entièrement nulle`, aucune image n'a été ajoutée au stack. Vérifiez la cohérence des dimensions et la configuration de la reprojection.

---

## License / Licence

**(English)**
This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

**(Français)**
Ce projet est sous licence GNU General Public License v3.0. Voir le fichier [LICENSE](LICENSE) pour les détails.

---

## Author / Auteur

*   **Tinystork**

## Acknowledgements / Crédits

### EN English version

While this project was primarily designed and developed by your humble Tinystork, it owes much to the contributions, inspiration, and tools provided by others. A heartfelt thank you goes out to:

**AI Assistants**  
Significant help with design, debugging, code generation, and complex concepts was provided by AI language models, including OpenAI's ChatGPT, Anthropic's Claude, DeepSeek, and Google's Gemini AI models. Their ability to quickly translate ideas into working code played a key role in overcoming challenges and speeding up development.

**The Seestar Community**  
Inspiration and motivation for this project came from discussions and shared enthusiasm within the Seestar user community. Thank you for your feedback, your shared experiences, and for showing that a tool like this was truly needed.

**Open Source Libraries & Their Developers**  
This software stands on the shoulders of giants — the developers of amazing open-source libraries. Special thanks to the teams behind:

- **NumPy** – Core numerical operations.  
- **OpenCV** (via `opencv-python` & `opencv-contrib-python`) – Image processing tasks: debayering, denoising, transformations, and more.  
- **Astropy** – FITS file handling and astronomy utilities.  
- **Astroalign** – The alignment engine at the heart of the stacker.  
- **Matplotlib** – For histogram visualization.  
- **Pillow** (PIL fork) – For loading, saving, and previewing images.  
- **Scikit-image** – Dependencies for alignment and other image analysis.  
- **Tqdm** – Smooth progress bars for terminal and logs.  
- **Psutil** *(optional)* – System monitoring for auto batch size tuning.  
- **CuPy** *(optional)* – Optional GPU acceleration for stacking.  
- **Python & Tkinter** – The foundation of the language and GUI.

Thank you to everyone whose work, directly or indirectly, made ZeSeestarStacker possible!

💫 Special thanks to Cheems, Astrobider, Clinton, Drinksalot, 550ml, Zcom et Qtopplings for their time, sharp eyes, and relentless beta testing — this project wouldn’t be half as solid without your help. You rock! 🌟
Your feedback helped shape features, squash bugs, and push the project further than I could alone.
Thanks for pointing out the weird stuff... and sometimes creating it too 😉

---

### 🇫🇷 Version française

Bien que ce projet ait été principalement conçu et développé par votre humble Tinystork, il doit énormément aux contributions, à l’inspiration et aux outils offerts par d’autres. Un grand merci à :

**Assistants IA**  
Une aide précieuse a été apportée pour la conception, le débogage, la génération de code et la compréhension de concepts complexes, grâce à des modèles de langage comme ChatGPT d’OpenAI, Claude d’Anthropic, DeepSeek, et Google Gemini . Leur capacité à transformer rapidement des idées en code fonctionnel a été déterminante dans les moments clés du développement.

**La communauté Seestar**  
Ce projet tire son inspiration et sa motivation des échanges passionnés au sein de la communauté des utilisateurs du Seestar. Merci pour vos retours, vos expériences partagées, et pour avoir souligné le besoin d’un outil comme celui-ci.

**Les bibliothèques open source & leurs développeurs**  
Ce logiciel repose sur les épaules de géants : les développeurs de bibliothèques open source exceptionnelles. Merci tout particulier aux équipes derrière :

- **NumPy** – Pour les opérations numériques de base.  
- **OpenCV** (via `opencv-python` & `opencv-contrib-python`) – Pour le traitement d’image : débayerisation, débruitage, transformations, etc.  
- **Astropy** – Pour la gestion des fichiers FITS et les utilitaires astronomiques.  
- **Astroalign** – Pour l’algorithme d’alignement au cœur du stacker.  
- **Matplotlib** – Pour l’affichage des histogrammes.  
- **Pillow** (fork de PIL) – Pour le chargement, la sauvegarde et l’aperçu des images.  
- **Scikit-image** – En tant que dépendance pour l’alignement et d’autres traitements.  
- **Tqdm** – Pour les barres de progression en terminal/logs.  
- **Psutil** *(optionnel)* – Pour la surveillance système et l’estimation automatique de la taille des lots.  
- **CuPy** *(optionnel)* – Pour l’accélération GPU facultative.  
- **Python & Tkinter** – Pour le langage et l’interface graphique.

Merci à toutes celles et ceux dont le travail, direct ou indirect, a rendu ZeSeestarStacker possible !

💫 Remerciements spéciaux à Cheems, Astrobider, Clinton, Drinksalot, 550ml, Zcom et Qtopplings pour leur temps, leur œil affûté et leurs tests acharnés — ce projet ne serait pas aussi solide sans votre aide. Vous déchirez ! 🌟
Vos retours ont permis de façonner les fonctionnalités, de traquer les bugs, et de faire avancer le projet bien plus loin que je ne l’aurais fait seul.
Merci d’avoir signalé les bizarreries… et parfois même de les avoir provoquées 😉
