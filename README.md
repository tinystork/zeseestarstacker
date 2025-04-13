# ZeSeestarStacker

ZeSeestarStacker is a Python-based tool designed to simplify and optimize the process of stacking a large number of light frames (images) often used in astrophotography and especially the seestar range of devices.

---

## Key Features
- **Batch Image Alignment**: Automatically aligns large quantities of light frames using `astroalign`.
- **Customizable Stacking Methods**:
  - **Mean Stacking**: Simple average stacking.
  - **Median Stacking**: Removes outliers by taking the median.
  - **Kappa-Sigma Clipping**: Removes outliers iteratively based on a sigma threshold.
  - **Winsorized Sigma Clipping**: A robust method that replaces outliers with boundary values.
- **Automatic Reference Frame Selection**:
  - Uses the sharpest and most contrast-rich frame as the reference by default.
  - Option to manually specify a reference frame.
- **Debayering**: Supports debayering for raw images with `GRBG` and `RGGB` Bayer patterns.
- **Memory Optimization**: Processes images in batches to handle large datasets efficiently.
- **Graphical User Interface (GUI)**: Built-in GUI for a user-friendly experience.

---

## Installation

### Prerequisites
- Python 3.8 or later
- Required Python packages (listed in `requirements.txt`):
  - `numpy`
  - `opencv-python`
  - `astropy`
  - `astroalign`
  - `tqdm`
  - `psutil`

### Installation Steps
1. Clone the Repository:
   ```bash
   git clone https://github.com/tinystork/zeseestarstacker.git
   cd zeseestarstacker
   ```

2. Install the Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Command Line Interface (CLI)
Run the script directly from the command line:
```bash
python align_and_stack.py
```

Follow the on-screen prompts to:
- Specify the input folder containing light frames (`.fit` or `.fits` files).
- Choose a stacking method (e.g., `mean`, `median`, `kappa-sigma`, `winsorized-sigma`).
- Optionally specify a manual reference frame.

### Graphical User Interface (GUI)
Run the GUI script:
```bash
python seestar_stacker_gui.py
```

#### GUI Features:
- Select input and output folders.
- Configure stacking methods and parameters.
- Monitor progress and estimated time remaining.
- View logs and status updates in real-time.

---

## Example Workflow
1. Collect light frames with your astrophotography equipment.
2. Place the `.fit` or `.fits` files into a folder (e.g., `lights/`).
3. Launch ZeSeestarStacker (GUI or CLI).
4. Select the folder containing your light frames.
5. Choose your desired stacking method and parameters.
6. Let the tool align, stack, and save the processed image to the output folder.

---

## Output
- **Aligned Images**: Saved in the `aligned_lights/` folder under the output directory.
- **Unaligned Images**: Any images that could not be aligned are saved in the `aligned_lights/unaligned/` folder.
- **Sub-stacks**: Intermediate stacked images for each batch, saved in the `sub_stacks/` folder.
- **Final Stacked Image**: The final processed image is saved in the root of the output directory.

---

## Contributing
Contributions are welcome! Feel free to:
- Submit issues or feature requests.
- Fork the repository and create pull requests.

---

## License
This project is licensed under the [GNU General Public License v3.0](LICENSE).

---

## Acknowledgments
- Built with Python and open-source libraries like `numpy`, `opencv-python`, `astropy`, `astroalign`, and `psutil`.
- Inspired by the needs of astrophotography enthusiasts to process large datasets efficiently.

---

## Contact
For questions or support, feel free to reach out via [GitHub Issues](https://github.com/tinystork/zeseestarstacker/issues).
