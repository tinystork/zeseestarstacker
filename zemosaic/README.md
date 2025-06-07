# 🌌 ZeMosaic

**ZeMosaic** is an open-source tool for assembling **large astronomical mosaics** from FITS images, with particular support for all-in-one sensors like the **Seestar S50**.

It was born out of a need from an astrophotography Discord community called the seestar collective stacking tens of **thousands of FITS images** into clean wide-field mosaics — a task where most existing tools struggled with scale, automation, or quality.

---

## 🚀 Key Features

- Astrometric alignment using **ASTAP**
- Smart tile grouping and automatic clustering
- Configurable stacking with:
  - **Noise-based weighting** (1/σ²)
  - **Kappa-Sigma** and **Winsorized** rejection
  - Radial feathering to blend tile borders
- Two mosaic assembly modes:
  - `Reproject & Coadd` (high quality, RAM-intensive)
  - `Incremental` (low memory, scalable)
- Stretch preview generation (ASIFits-style)
- GUI built with **Tkinter**, fully translatable (EN/FR)
- Flexible FITS export with configurable `axis_order` (default `HWC`) and
  proper `BSCALE`/`BZERO` for float images
- Option to save the final mosaic as 16-bit integer FITS

---

## 📷 Requirements

### Mandatory:

- Python ≥ 3.9  
- [ASTAP](https://www.hnsky.org/astap.htm) installed with G17/H17 star catalogs

### Recommended Python packages:

```bash
pip install numpy astropy reproject opencv-python photutils scipy psutil
No versions are pinned, but ZeMosaic is tested on Python 3.11+.

🧠 Inspired by PixInsight
ZeMosaic draws strong inspiration from the image integration strategies of PixInsight, developed by Juan Conejero at Pleiades Astrophoto.

Specifically, the implementations of:

Noise Variance Weighting (1/σ²)

Kappa-Sigma and Winsorized Rejection

Radial feather blending

...are adapted from methods described in:

📖 PixInsight 1.6.1 – New ImageIntegration Features
Juan Conejero, 2010
Forum thread

🙏 We gratefully acknowledge Juan Conejero's contributions to astronomical image processing.

🛠 Dependencies
ZeMosaic uses several powerful open-source Python libraries:

numpy and scipy for numerical processing

astropy for FITS I/O and WCS handling

reproject for celestial reprojection

opencv-python for debayering

photutils for source detection and background estimation

psutil for memory monitoring

tkinter for the graphical user interface

📦 Installation & Usage
1. 🔧 Install Python dependencies
If you have a local clone of the repository, make sure you're in the project folder, then run:

pip install -r requirements.txt
💡 No versions are pinned in requirements.txt to maintain flexibility. ZeMosaic is tested with Python 3.11+.

If you prefer to install manually:

pip install numpy astropy reproject opencv-python photutils scipy psutil

2. 🚀 Launch ZeMosaic
Once the dependencies are installed:
python run_zemosaic.py

The GUI will open. From there:

Select your input folder (with raw FITS images)

Choose your output folder

Configure ASTAP paths and options

Adjust stacking & mosaic settings

Click "Start Hierarchical Mosaic"

📁 Requirements Summary
✅ Python 3.9 or newer

✅ ASTAP installed + star catalogs (D50 or H18)

✅ FITS images (ideally calibrated, debayered or raw from Seestar)

🖥️ How to Run
After installing Python and dependencies:

python run_zemosaic.py
Use the GUI to:

Choose your input/output folders

Configure ASTAP paths

Select stacking and assembly options

Click Start Hierarchical Mosaic

🔧 Build & Compilation (Windows) / Compilation (Windows)
🇬🇧 Instructions (English)
To build the standalone executable version of ZeMosaic, follow these steps:

Install Python 3.13 from python.org

Create and activate a virtual environment (if not already done):

powershell
Copier
Modifier
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Build the .exe by running:

powershell
Copier
Modifier
compile\build_zemosaic.bat
The final executable will be created in dist/zemosaic.exe.

✅ Translations (locales/*.json) and application icons (icon/zemosaic.ico) are automatically included.

🇫🇷 Instructions (Français)
Pour créer l'exécutable autonome de ZeMosaic, suivez ces étapes :

Installez Python 3.13 depuis python.org

Créez et activez un environnement virtuel (si ce n’est pas déjà fait) :

powershell
Copier
Modifier
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Lancez la compilation de l’exécutable avec :

powershell
Copier
Modifier
compile\build_zemosaic.bat
L’exécutable final se trouvera dans dist/zemosaic.exe.

✅ Les fichiers de traduction (locales/*.json) et les icônes (icon/zemosaic.ico) sont inclus automatiquement.




🧪 Troubleshooting
If astrometric solving fails:

Check ASTAP path and data catalogs

Ensure your images contain enough stars

Use a Search Radius of ~3.0°

Watch zemosaic_worker.log for full tracebacks

📎 License
ZeMosaic is licensed under GPLv3 — feel free to use, adapt, and contribute.

🤝 Contributions
Feature requests, bug reports, and pull requests are welcome!
Please include log files and test data if possible when reporting issues.

🌠 Happy mosaicking!