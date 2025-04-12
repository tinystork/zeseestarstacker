import os
import numpy as np
from astropy.io import fits
import cv2
import astroalign as aa
from tqdm import tqdm
import warnings
import gc

warnings.filterwarnings("ignore", category=FutureWarning)


def load_and_validate_fits(path):
    """
    Load and validate FITS files, ensuring they are 2D or 3D images.
    """
    data = fits.getdata(path)
    data = np.squeeze(data).astype(np.float32)
    if data.ndim not in [2, 3]:
        raise ValueError("L'image doit être 2D (HxW) ou 3D (HxWx3)")
    return data


def debayer_image(img, bayer_pattern="GRBG"):
    """
    Convert a raw Bayer image to RGB.
    """
    img_uint16 = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)

    if bayer_pattern == "GRBG":
        color_img = cv2.cvtColor(img_uint16, cv2.COLOR_BayerGR2RGB)
    elif bayer_pattern == "RGGB":
        color_img = cv2.cvtColor(img_uint16, cv2.COLOR_BayerRG2RGB)
    else:
        raise ValueError(f"Motif Bayer {bayer_pattern} non supporté")

    return color_img.astype(np.float32)


def align_seestar_images_batch(input_folder, bayer_pattern="GRBG", batch_size=10, manual_reference_path=None):
    """
    Align Seestar images in batches with an optional manual reference image.
    """
    # Ensure batch_size is never zero or negative
    if batch_size <= 0:
        batch_size = 10  # Default value

    output_folder = os.path.join(input_folder, "aligned_lights")
    os.makedirs(output_folder, exist_ok=True)
    
    # Créer un dossier séparé pour les images non alignées
    unaligned_folder = os.path.join(output_folder, "unaligned")
    os.makedirs(unaligned_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.fit', '.fits'))]
    if not files:
        print("❌ Aucun fichier .fit/.fits trouvé")
        return output_folder  # Retourner quand même le dossier de sortie

    print(f"\n🔍 Analyse de {len(files)} images...")

    fixed_reference_image = None
    fixed_reference_header = None

    # Tentative de chargement de l'image de référence manuelle si fournie
    if manual_reference_path:
        try:
            print(f"\n📌 Chargement de l'image de référence manuelle : {manual_reference_path}")
            fixed_reference_image = load_and_validate_fits(manual_reference_path)
            fixed_reference_header = fits.getheader(manual_reference_path)
            
            # Appliquer debayer sur l'image de référence si c'est une image brute
            if fixed_reference_image.ndim == 2:
                fixed_reference_image = debayer_image(fixed_reference_image, bayer_pattern)
            elif fixed_reference_image.ndim == 3 and fixed_reference_image.shape[0] == 3:
                # Pour les images 3D avec la première dimension comme canal
                fixed_reference_image = np.moveaxis(fixed_reference_image, 0, -1)  # Convert to HxWx3
                
            fixed_reference_image = cv2.normalize(fixed_reference_image, None, 0, 65535, cv2.NORM_MINMAX)
            print(f"✅ Image de référence chargée: dimensions: {fixed_reference_image.shape}")
        except Exception as e:
            print(f"❌ Erreur lors du chargement de l'image de référence manuelle: {e}")
            fixed_reference_image = None

    # Pré-chargement des images pour trouver la meilleure référence si aucune référence manuelle n'est fournie
    if fixed_reference_image is None:
        print("\n⚙️ Recherche de la meilleure image de référence...")
        sample_images = []
        sample_headers = []
        sample_files = []
        
        for f in tqdm(files[:min(batch_size*2, len(files))], desc="Analyse des images"):
            try:
                img_path = os.path.join(input_folder, f)
                img = load_and_validate_fits(img_path)
                hdr = fits.getheader(img_path)
                
                # S'assurer que l'image a une variance suffisante
                if np.std(img) > 5:
                    # Convertir en couleur si nécessaire
                    if img.ndim == 2:
                        img = debayer_image(img, bayer_pattern)
                    elif img.ndim == 3 and img.shape[0] == 3:
                        # Pour les images 3D avec la première dimension comme canal
                        img = np.moveaxis(img, 0, -1)  # Convert to HxWx3
                    
                    img = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX)
                    
                    sample_images.append(img)
                    sample_headers.append(hdr)
                    sample_files.append(f)
                else:
                    print(f"⚠️ Image ignorée (faible variance): {f}")
            except Exception as e:
                print(f"❌ Erreur lors de l'analyse de {f}: {e}")
        
        if sample_images:
            # Sélectionner l'image avec le meilleur contraste (médiane la plus élevée)
            medians = [np.median(img) for img in sample_images]
            ref_idx = np.argmax(medians)
            fixed_reference_image = sample_images[ref_idx]
            fixed_reference_header = sample_headers[ref_idx]
            print(f"\n⭐ Référence utilisée: {sample_files[ref_idx]}")
            
            # Sauvegarder l'image de référence
            ref_output_path = os.path.join(output_folder, "reference_image.fit")
            ref_data = np.moveaxis(fixed_reference_image, -1, 0).astype(np.uint16)  # HxWx3 to 3xHxW
            
            new_header = fixed_reference_header.copy()
            new_header['NAXIS'] = 3
            new_header['NAXIS1'] = fixed_reference_image.shape[1]
            new_header['NAXIS2'] = fixed_reference_image.shape[0]
            new_header['NAXIS3'] = 3
            new_header['BITPIX'] = 16
            new_header.set('CTYPE3', 'RGB', 'Couleurs RGB')
            
            fits.writeto(ref_output_path, ref_data, new_header, overwrite=True)
            print(f"📁 Image de référence sauvegardée: {ref_output_path}")
        else:
            print("❌ Impossible de trouver une image de référence valide.")
            return output_folder

    # Traitement par lots
    for batch_start in range(0, len(files), batch_size):
        batch_files = files[batch_start:batch_start + batch_size]
        print(f"\n🚀 Traitement du lot {batch_start // batch_size + 1} (images {batch_start + 1} à {batch_start + len(batch_files)})...")
        
        images = []
        headers = []
        valid_files = []
        
        # Chargement des images du lot
        for f in tqdm(batch_files, desc="Chargement"):
            try:
                img_path = os.path.join(input_folder, f)
                img = load_and_validate_fits(img_path)
                
                # Vérifier la qualité de l'image
                if np.std(img) > 5:
                    if img.ndim == 2:
                        img = debayer_image(img, bayer_pattern)
                    elif img.ndim == 3 and img.shape[0] == 3:
                        img = np.moveaxis(img, 0, -1)  # Convert to HxWx3
                    
                    img = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX)
                    
                    images.append(img)
                    headers.append(fits.getheader(img_path))
                    valid_files.append(f)
                else:
                    print(f"Rejetée (qualité insuffisante): {f}")
            except Exception as e:
                print(f"⚠️ {f}: {str(e)}")
        
        if len(images) < 1:
            print("❌ Aucune image valide dans ce lot, ignoré.")
            continue
            
        # Alignement des images du lot
        for i, (img, hdr, fname) in enumerate(zip(tqdm(images, desc="Alignement"), headers, valid_files)):
            try:
                # Alignement canal par canal pour les images couleur
                if img.ndim == 3:
                    aligned_channels = []
                    for c in range(3):
                        # Normalisation pour l'alignement
                        img_norm = cv2.normalize(img[:, :, c], None, 0, 1, cv2.NORM_MINMAX)
                        ref_norm = cv2.normalize(fixed_reference_image[:, :, c], None, 0, 1, cv2.NORM_MINMAX)
                        
                        aligned_channel, _ = aa.register(img_norm, ref_norm)
                        aligned_channels.append(aligned_channel)
                    
                    # Recombiner les canaux
                    aligned_img = np.stack(aligned_channels, axis=-1)
                    aligned_img = cv2.normalize(aligned_img, None, 0, 65535, cv2.NORM_MINMAX)
                else:
                    # Cas d'une image en niveaux de gris
                    aligned_img, _ = aa.register(img, fixed_reference_image)
                    aligned_img = np.stack((aligned_img,) * 3, axis=-1)
                
                # Conversion au format FITS pour l'enregistrement (HxWx3 -> 3xHxW)
                color_cube = np.moveaxis(aligned_img, -1, 0).astype(np.uint16)
                
                # Mettre à jour l'en-tête
                new_header = hdr.copy()
                new_header['NAXIS'] = 3
                new_header['NAXIS1'] = aligned_img.shape[1]
                new_header['NAXIS2'] = aligned_img.shape[0]
                new_header['NAXIS3'] = 3
                new_header['BITPIX'] = 16
                new_header.set('CTYPE3', 'RGB', 'Couleurs RGB')
                
                # Enregistrement de l'image alignée
                out_path = os.path.join(output_folder, f"aligned_{batch_start + i:04}.fit")
                fits.writeto(out_path, color_cube, new_header, overwrite=True)
            except Exception as e:
                print(f"❌ Échec de l'alignement pour l'image {fname}: {str(e)}")
                
                # Copier l'image non alignée dans le dossier séparé
                try:
                    original_path = os.path.join(input_folder, fname)
                    out_path = os.path.join(unaligned_folder, f"unaligned_{fname}")
                    with open(original_path, 'rb') as src, open(out_path, 'wb') as dst:
                        dst.write(src.read())
                    print(f"⚠️ Image non alignée sauvegardée: {out_path}")
                except Exception as copy_err:
                    print(f"❌ Impossible de copier l'image originale: {copy_err}")
        
        # Libérer la mémoire après le traitement du lot
        del images
        del headers
        gc.collect()

    print(f"\n✅ Toutes les images alignées en couleur ont été sauvegardées dans: {output_folder}")
    return output_folder
    

if __name__ == "__main__":
    input_path = input("📂 Entrez le chemin du dossier contenant les images FITS : ").strip('"\' ')
    reference_path = input("📌 Entrez le chemin de l'image de référence (laisser vide pour sélection dynamique) : ").strip('"\' ')
    batch_size = int(input("🛠️ Entrez la taille du lot (exemple : 10) : ") or 10)
    align_seestar_images_batch(input_path, batch_size=batch_size, manual_reference_path=reference_path)
    input("\nAppuyez sur Entrée pour quitter...")