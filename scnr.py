import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from astropy.io import fits
from matplotlib import pyplot as plt

def load_fits_image(path):
    """
    Charge une image FITS et la convertit en numpy array.
    """
    data = fits.getdata(path)
    if data is None or data.size == 0:
        raise ValueError(f"Erreur de chargement de l'image FITS. Le fichier semble vide : {path}")
    return np.array(data, dtype=np.float32)

def debayer_image(img, bayer_pattern="GRBG"):
    img_uint16 = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)
    if bayer_pattern == "GRBG":
        color_img = cv2.cvtColor(img_uint16, cv2.COLOR_BayerGR2RGB)
    elif bayer_pattern == "RGGB":
        color_img = cv2.cvtColor(img_uint16, cv2.COLOR_BayerRG2RGB)
    else:
        raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")
    return color_img.astype(np.float32)

def balance_white_simple(img):
    r, g, b = cv2.split(img)
    r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
    avg = (r_mean + g_mean + b_mean) / 3.0
    r *= avg / r_mean
    g *= avg / g_mean
    b *= avg / b_mean
    return cv2.merge([r, g, b])

def remove_green_noise(img, strength=1.0):
    r, g, b = cv2.split(img)
    g_reduced = g - strength * np.minimum(g - r, g - b)
    g_reduced = np.clip(g_reduced, 0, None)
    return cv2.merge([r, g_reduced, b])

def process_image(image_path, bayer_pattern="GRBG", scnr_strength=1.0):
    if image_path.lower().endswith(".fit") or image_path.lower().endswith(".fits"):
        raw = load_fits_image(image_path)
    else:
        raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    print(f"Image chargée : {raw.shape}")

    debayered = debayer_image(raw, bayer_pattern)
    print(f"Image après débayer : {debayered.shape}")

    white_balanced = balance_white_simple(debayered)
    print(f"Image après balance des blancs : {white_balanced.shape}")

    green_removed = remove_green_noise(white_balanced, scnr_strength)
    print(f"Image après SCNR : {green_removed.shape}")

    return debayered, white_balanced, green_removed

def save_fits_image(image, output_path, header=None, overwrite=True):
    if header is None:
        header = fits.Header()

    if image.ndim == 3:
        image_fits = np.moveaxis(image, -1, 0)  # HxWx3 → 3xHxW
        header['NAXIS'] = 3
        header['NAXIS1'] = image.shape[1]
        header['NAXIS2'] = image.shape[0]
        header['NAXIS3'] = 3
        header['BITPIX'] = 16
        header.set('CTYPE3', 'RGB', 'Couleurs RGB')
    else:
        image_fits = image
        header['NAXIS'] = 2
        header['NAXIS1'] = image.shape[1]
        header['NAXIS2'] = image.shape[0]
        header['BITPIX'] = 16

    image_fits = cv2.normalize(image_fits, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)
    fits.writeto(output_path, image_fits, header, overwrite=overwrite)
    print(f"Image enregistrée sous : {output_path}")

def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Choisir une image (FITS, PNG, JPG, TIF)", filetypes=[("Images", "*.fits;*.fit;*.png;*.jpg;*.tif")])

    if file_path:
        debayered, white_balanced, green_removed = process_image(file_path)
        
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].imshow(np.clip(debayered / 65535, 0, 1))
        axs[0].set_title("Débayerisé")
        axs[1].imshow(np.clip(white_balanced / 65535, 0, 1))
        axs[1].set_title("Balance des blancs")
        axs[2].imshow(np.clip(green_removed / 65535, 0, 1))
        axs[2].set_title("SCNR (vert réduit)")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

        save_path = filedialog.asksaveasfilename(title="Enregistrer l'image", defaultextension=".png", filetypes=[("Images", "*.fits;*.png;*.jpg;*.tif")])
        if save_path:
            if save_path.lower().endswith(".fits"):
                save_fits_image(green_removed, save_path)
            else:
                cv2.imwrite(save_path, np.clip(green_removed / 65535, 0, 255).astype(np.uint8))
    else:
        print("Aucun fichier sélectionné.")

if __name__ == "__main__":
    main()
