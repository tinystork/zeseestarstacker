import os
import numpy as np
from astropy.io import fits
import cv2
import astroalign as aa
from tqdm import tqdm
import warnings
import gc
import psutil
from datetime import datetime, timedelta
import time
import shutil

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


def kappa_sigma_stack(images, kappa=2.5, iterations=3):
    """Stack images using the kappa-sigma clipping algorithm."""
    if len(images) < 3:
        return np.mean(images, axis=0)

    # Vérifier si les images sont en RGB ou monochrome
    is_rgb = len(images[0].shape) == 3 and images[0].shape[2] == 3
    
    if is_rgb:
        stack = np.zeros_like(images[0], dtype=np.float32)
        for c in range(images[0].shape[2]):
            data = np.array([img[..., c] for img in images])
            mask = np.ones_like(data, dtype=bool)
            mean = np.mean(data, axis=0)

            for _ in range(iterations):
                std = np.std(data, axis=0)
                new_mask = np.abs(data - mean) <= (kappa * std)
                mask = new_mask
                non_zero_count = np.sum(mask, axis=0)
                non_zero_count[non_zero_count == 0] = 1
                mean = np.sum(data * mask, axis=0) / non_zero_count

            stack[..., c] = mean
    else:
        # Pour les images monochromes
        data = np.array(images)
        mask = np.ones_like(data, dtype=bool)
        mean = np.mean(data, axis=0)

        for _ in range(iterations):
            std = np.std(data, axis=0)
            new_mask = np.abs(data - mean) <= (kappa * std)
            mask = new_mask
            non_zero_count = np.sum(mask, axis=0)
            non_zero_count[non_zero_count == 0] = 1
            mean = np.sum(data * mask, axis=0) / non_zero_count
        
        stack = mean
        
    return stack


def winsorized_sigma_stack(images, kappa=2.5, iterations=3):
    """
    Stack images using Winsorized Sigma Clipping.
    
    This method replaces outliers with the values at the boundaries instead of 
    completely removing them, which can provide better results in some cases.
    """
    if len(images) < 3:
        return np.mean(images, axis=0)

    # Vérifier si les images sont en RGB ou monochrome
    is_rgb = len(images[0].shape) == 3 and images[0].shape[2] == 3
    
    if is_rgb:
        stack = np.zeros_like(images[0], dtype=np.float32)
        for c in range(images[0].shape[2]):
            data = np.array([img[..., c] for img in images])
            
            for _ in range(iterations):
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                
                lower_bound = mean - kappa * std
                upper_bound = mean + kappa * std
                
                # Winsorize: replace values outside bounds with the bounds
                data = np.maximum(data, lower_bound)
                data = np.minimum(data, upper_bound)
            
            # Final mean after winsorization
            stack[..., c] = np.mean(data, axis=0)
    else:
        # Pour les images monochromes
        data = np.array(images)
        
        for _ in range(iterations):
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            
            lower_bound = mean - kappa * std
            upper_bound = mean + kappa * std
            
            # Winsorize: replace values outside bounds with the bounds
            data = np.maximum(data, lower_bound)
            data = np.minimum(data, upper_bound)
        
        # Final mean after winsorization
        stack = np.mean(data, axis=0)
    
    return stack


def format_exposure_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    return str(timedelta(seconds=int(seconds)))


def estimate_remaining_time(start_time, files_done, total_files):
    """
    Estimate remaining processing time based on elapsed time and progress.
    """
    if files_done == 0:
        return "Calcul en cours..."
    
    elapsed_time = time.time() - start_time
    time_per_file = elapsed_time / files_done
    remaining_files = total_files - files_done
    remaining_seconds = time_per_file * remaining_files
    
    # Format as HH:MM:SS
    return format_exposure_time(remaining_seconds)


def stack_batch(batch_images, batch_headers, stacking_mode="kappa-sigma", kappa=2.5, max_iterations=3):
    """
    Stack a batch of images with the specified method.
    
    Parameters:
        batch_images (list): List of loaded image data arrays
        batch_headers (list): List of FITS headers
        stacking_mode (str): Method to use for stacking
        kappa (float): Kappa value for sigma-clipping methods
        max_iterations (int): Number of iterations for iterative methods
        
    Returns:
        tuple: (stacked_image, combined_header)
    """
    if not batch_images:
        return None, None
        
    # Apply stacking method
    if stacking_mode == "mean":
        batch_stack = np.mean(batch_images, axis=0)
    elif stacking_mode == "median":
        batch_stack = np.median(batch_images, axis=0)
    elif stacking_mode == "winsorized-sigma":
        batch_stack = winsorized_sigma_stack(batch_images, kappa, max_iterations)
    else:  # Default to kappa-sigma
        batch_stack = kappa_sigma_stack(batch_images, kappa, max_iterations)
    
    # Combine header information
    combined_header = batch_headers[0].copy() if batch_headers else fits.Header()
    combined_header['STACKING'] = stacking_mode, 'Method of stacking'
    combined_header['NUMIMGS'] = len(batch_images), 'Number of stacked images'
    
    # Calculate total exposure time if available
    total_exposure_time = 0
    for header in batch_headers:
        if "EXPTIME" in header:
            total_exposure_time += header["EXPTIME"]
    
    if total_exposure_time > 0:
        combined_header['EXPTIME'] = total_exposure_time, 'Total equivalent exposure time'
    
    combined_header['DATE-STK'] = datetime.now().isoformat(), 'Date of stacking'
    
    return batch_stack, combined_header


def align_and_stack_seestar_images(
    input_folder, 
    output_folder=None,
    stacking_mode="kappa-sigma", 
    kappa=2.5, 
    max_iterations=3,
    bayer_pattern="GRBG", 
    batch_size=10, 
    manual_reference_path=None,
    progress_callback=None
):
    """
    Align and stack Seestar images in batches with continuous optimization.
    
    Parameters:
        input_folder (str): Path to the folder containing input .fit/.fits files.
        output_folder (str): Path to save output files. Defaults to input_folder/processed
        stacking_mode (str): Stacking method ('mean', 'median', 'kappa-sigma', 'winsorized-sigma')
        kappa (float): Kappa value for sigma-clipping methods
        max_iterations (int): Number of iterations for iterative methods
        bayer_pattern (str): Bayer pattern for debayering (default: "GRBG").
        batch_size (int): Number of images to process per batch (default: 10).
        manual_reference_path (str): Optional path to a manually selected reference image.
        progress_callback (function): Optional callback for progress updates
    """
    # Setup output folders
    if output_folder is None:
        output_folder = os.path.join(input_folder, "processed")
        
    # Create necessary folders
    os.makedirs(output_folder, exist_ok=True)
    aligned_folder = os.path.join(output_folder, "aligned_lights")
    os.makedirs(aligned_folder, exist_ok=True)
    unaligned_folder = os.path.join(aligned_folder, "unaligned")
    os.makedirs(unaligned_folder, exist_ok=True)
    substack_folder = os.path.join(output_folder, "sub_stacks")
    os.makedirs(substack_folder, exist_ok=True)
    
    # Helper for progress updates
    def update_progress(message, progress=None):
        if progress_callback:
            progress_callback(message, progress)
        else:
            print(message)

    # Start timing
    processing_start_time = time.time()
    
    # Find all FITS files in the input folder
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.fit', '.fits'))]
    if not files:
        update_progress("❌ Aucun fichier .fit/.fits trouvé")
        return None
    
    total_files = len(files)
    update_progress(f"\n🔍 Analyse de {total_files} images...")

    # Ensure batch_size is never zero or negative
    if batch_size <= 0:
        batch_size = 10  # Default value
    
    # Initialize variables for reference and batches
    fixed_reference_image = None
    fixed_reference_header = None
    batch_stacks = []
    batch_stack_headers = []
    total_exposure_time = 0
    processed_files = 0

    # Try to load manual reference if provided
    if manual_reference_path:
        try:
            update_progress(f"\n📌 Chargement de l'image de référence manuelle : {manual_reference_path}")
            fixed_reference_image = load_and_validate_fits(manual_reference_path)
            fixed_reference_header = fits.getheader(manual_reference_path)
            
            # Apply debayer to reference image if it's a raw image
            if fixed_reference_image.ndim == 2:
                fixed_reference_image = debayer_image(fixed_reference_image, bayer_pattern)
            elif fixed_reference_image.ndim == 3 and fixed_reference_image.shape[0] == 3:
                # For 3D images with first dimension as channel
                fixed_reference_image = np.moveaxis(fixed_reference_image, 0, -1)  # Convert to HxWx3
                
            fixed_reference_image = cv2.normalize(fixed_reference_image, None, 0, 65535, cv2.NORM_MINMAX)
            update_progress(f"✅ Image de référence chargée: dimensions: {fixed_reference_image.shape}")
        except Exception as e:
            update_progress(f"❌ Erreur lors du chargement de l'image de référence manuelle: {e}")
            fixed_reference_image = None

    # Find best reference image if not manually provided
    if fixed_reference_image is None:
        update_progress("\n⚙️ Recherche de la meilleure image de référence...")
        sample_images = []
        sample_headers = []
        sample_files = []
        
        for f in tqdm(files[:min(batch_size*2, len(files))], desc="Analyse des images"):
            try:
                img_path = os.path.join(input_folder, f)
                img = load_and_validate_fits(img_path)
                hdr = fits.getheader(img_path)
                
                # Ensure image has sufficient variance
                if np.std(img) > 5:
                    # Convert to color if needed
                    if img.ndim == 2:
                        img = debayer_image(img, bayer_pattern)
                    elif img.ndim == 3 and img.shape[0] == 3:
                        # For 3D images with first dimension as channel
                        img = np.moveaxis(img, 0, -1)  # Convert to HxWx3
                    
                    img = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX)
                    
                    sample_images.append(img)
                    sample_headers.append(hdr)
                    sample_files.append(f)
                else:
                    update_progress(f"⚠️ Image ignorée (faible variance): {f}")
            except Exception as e:
                update_progress(f"❌ Erreur lors de l'analyse de {f}: {e}")
        
        if sample_images:
            # Select image with best contrast (highest median)
            medians = [np.median(img) for img in sample_images]
            ref_idx = np.argmax(medians)
            fixed_reference_image = sample_images[ref_idx]
            fixed_reference_header = sample_headers[ref_idx]
            update_progress(f"\n⭐ Référence utilisée: {sample_files[ref_idx]}")
            
            # Save reference image
            ref_output_path = os.path.join(aligned_folder, "reference_image.fit")
            ref_data = np.moveaxis(fixed_reference_image, -1, 0).astype(np.uint16)  # HxWx3 to 3xHxW
            
            new_header = fixed_reference_header.copy()
            new_header['NAXIS'] = 3
            new_header['NAXIS1'] = fixed_reference_image.shape[1]
            new_header['NAXIS2'] = fixed_reference_image.shape[0]
            new_header['NAXIS3'] = 3
            new_header['BITPIX'] = 16
            new_header.set('CTYPE3', 'RGB', 'Couleurs RGB')
            
            fits.writeto(ref_output_path, ref_data, new_header, overwrite=True)
            update_progress(f"📁 Image de référence sauvegardée: {ref_output_path}")
        else:
            update_progress("❌ Impossible de trouver une image de référence valide.")
            return None

    # Process images in batches
    for batch_start in range(0, len(files), batch_size):
        # Get batch files
        batch_files = files[batch_start:batch_start + batch_size]
        update_progress(f"\n🚀 Traitement du lot {batch_start // batch_size + 1} (images {batch_start + 1} à {batch_start + len(batch_files)})...")
        
        # Variables for this batch
        aligned_images = []
        aligned_headers = []
        aligned_paths = []
        unaligned_paths = []
        
        # Process each image in the batch
        for i, f in enumerate(tqdm(batch_files, desc="Alignement")):
            try:
                img_path = os.path.join(input_folder, f)
                img = load_and_validate_fits(img_path)
                hdr = fits.getheader(img_path)
                
                # Vérifier la qualité de l'image
                if np.std(img) > 5:
                    if img.ndim == 2:
                        img = debayer_image(img, bayer_pattern)
                    elif img.ndim == 3 and img.shape[0] == 3:
                        img = np.moveaxis(img, 0, -1)  # Convert to HxWx3
                    
                    img = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX)
                    
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
                    out_path = os.path.join(aligned_folder, f"aligned_{batch_start + i:04}.fit")
                    fits.writeto(out_path, color_cube, new_header, overwrite=True)
                    
                    # Store in memory for stacking
                    aligned_images.append(aligned_img)
                    aligned_headers.append(new_header)
                    aligned_paths.append(out_path)
                    
                    # Accumulate exposure time
                    if "EXPTIME" in hdr:
                        total_exposure_time += hdr["EXPTIME"]
                else:
                    update_progress(f"Rejetée (qualité insuffisante): {f}")
            except Exception as e:
                update_progress(f"❌ Échec de l'alignement pour l'image {f}: {str(e)}")
                
                # Copy unaligned image to separate folder
                try:
                    original_path = os.path.join(input_folder, f)
                    out_path = os.path.join(unaligned_folder, f"unaligned_{f}")
                    with open(original_path, 'rb') as src, open(out_path, 'wb') as dst:
                        dst.write(src.read())
                    update_progress(f"⚠️ Image non alignée sauvegardée: {out_path}")
                    unaligned_paths.append(out_path)
                except Exception as copy_err:
                    update_progress(f"❌ Impossible de copier l'image originale: {copy_err}")
            
            # Update progress
            processed_files += 1
            progress_percent = (processed_files / total_files) * 100
            remaining_time = estimate_remaining_time(processing_start_time, processed_files, total_files)
            update_progress(
                f"Traitement de {f} ({processed_files}/{total_files}) - "
                f"Temps restant estimé: {remaining_time}", 
                progress=progress_percent
            )
        
        # Stack aligned images from this batch if there are any
        if aligned_images:
            update_progress(f"⚙️ Création du sous-empilement pour le lot {batch_start // batch_size + 1}...")
            batch_stack, batch_header = stack_batch(
                aligned_images, 
                aligned_headers,
                stacking_mode=stacking_mode,
                kappa=kappa,
                max_iterations=max_iterations
            )
            
            if batch_stack is not None:
                # Save sub-stack
                substack_path = os.path.join(substack_folder, f"sub_stack_{batch_start // batch_size + 1:03d}.fit")
                
                # Convert to FITS format (HxWx3 -> 3xHxW)
                if batch_stack.ndim == 3 and batch_stack.shape[2] == 3:
                    batch_stack_fits = np.moveaxis(batch_stack, -1, 0).astype(np.float32)
                else:
                    batch_stack_fits = batch_stack.astype(np.float32)
                
                fits.writeto(substack_path, batch_stack_fits, batch_header, overwrite=True)
                update_progress(f"✅ Sous-empilement sauvegardé: {substack_path}")
                
                # Add to list of sub-stacks for final stacking
                batch_stacks.append(batch_stack)
                batch_stack_headers.append(batch_header)
                
                # Now that we have a sub-stack, delete the individual aligned files to save space
                update_progress("🧹 Suppression des fichiers alignés individuels...")
                for path in aligned_paths:
                    try:
                        os.remove(path)
                    except Exception as e:
                        update_progress(f"⚠️ Impossible de supprimer {path}: {e}")
        
        # Delete unaligned files to save space
        if unaligned_paths:
            update_progress("🧹 Suppression des fichiers non alignés...")
            for path in unaligned_paths:
                try:
                    os.remove(path)
                except Exception as e:
                    update_progress(f"⚠️ Impossible de supprimer {path}: {e}")
        
        # Clear memory
        gc.collect()
    
    # Check if we have any sub-stacks to create the final stack
    if not batch_stacks:
        update_progress("❌ Aucun sous-empilement créé, impossible de générer l'empilement final.")
        return None
    
    # Create final stack from sub-stacks
    update_progress("\n⭐ Création de l'empilement final...")
    
    # Stack all sub-stacks together
    final_stack, final_header = stack_batch(
        batch_stacks, 
        batch_stack_headers,
        stacking_mode=stacking_mode,
        kappa=kappa,
        max_iterations=max_iterations
    )
    
    if final_stack is None:
        update_progress("❌ Échec de la création de l'empilement final.")
        return None
    
    # Save final stack
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_path = os.path.join(output_folder, f"FINAL_STACK_{stacking_mode}_{timestamp}.fit")
    
    # Convert to FITS format (HxWx3 -> 3xHxW) if needed
    if final_stack.ndim == 3 and final_stack.shape[2] == 3:
        final_stack_fits = np.moveaxis(final_stack, -1, 0).astype(np.float32)
    else:
        final_stack_fits = final_stack.astype(np.float32)
    
    # Update final header
    final_header['NUMIMGS'] = total_files, 'Total number of input images'
    final_header['EXPTIME'] = total_exposure_time, 'Total equivalent exposure time'
    final_header['DATE-STK'] = datetime.now().isoformat(), 'Date of stacking'
    
    fits.writeto(final_path, final_stack_fits, final_header, overwrite=True)
    
    # Calculate total processing time
    total_time = time.time() - processing_start_time
    formatted_proc_time = format_exposure_time(total_time)
    
    update_progress(f"\n✅ Traitement terminé avec succès !")
    update_progress(f"📁 Empilement final: {final_path}")
    update_progress(f"📊 Nombre de fichiers traités: {total_files}")
    update_progress(f"⏱️ Durée totale du traitement: {formatted_proc_time}")
    if total_exposure_time > 0:
        formatted_exp_time = format_exposure_time(total_exposure_time)
        update_progress(f"⏱️ Temps d'exposition total équivalent: {formatted_exp_time}")
    
    # Clean up unaligned folder if empty
    try:
        if not os.listdir(unaligned_folder):
            os.rmdir(unaligned_folder)
            update_progress("🧹 Dossier 'unaligned' supprimé (vide)")
        
        # Clean up aligned folder if empty
        if not os.listdir(aligned_folder) or (len(os.listdir(aligned_folder)) == 1 and os.path.exists(os.path.join(aligned_folder, "reference_image.fit"))):
            # Just keep the reference image if needed
            ref_path = os.path.join(aligned_folder, "reference_image.fit")
            if os.path.exists(ref_path):
                # Move reference to output folder
                new_ref_path = os.path.join(output_folder, "reference_image.fit")
                shutil.move(ref_path, new_ref_path)
            
            # Remove aligned folder
            if os.path.exists(aligned_folder):
                shutil.rmtree(aligned_folder)
                update_progress("🧹 Dossier 'aligned_lights' supprimé (vide)")
    except Exception as e:
        update_progress(f"⚠️ Nettoyage des dossiers: {e}")
    
    return final_path


if __name__ == "__main__":
    input_path = input("📂 Entrez le chemin du dossier contenant les images FITS : ").strip('"\' ')
    reference_path = input("📌 Entrez le chemin de l'image de référence (laisser vide pour sélection dynamique) : ").strip('"\' ')
    stacking_method = input("📋 Choisissez la méthode d'empilement (mean, median, kappa-sigma, winsorized-sigma) [kappa-sigma] : ").strip() or "kappa-sigma"
    batch_size = int(input("🛠️ Entrez la taille du lot (exemple : 10) : ") or 10)
    
    if stacking_method in ["kappa-sigma", "winsorized-sigma"]:
        kappa = float(input("📊 Entrez la valeur de kappa (défaut: 2.5) : ") or 2.5)
        iterations = int(input("🔄 Entrez le nombre d'itérations (défaut: 3) : ") or 3)
    else:
        kappa = 2.5
        iterations = 3
    
    align_and_stack_seestar_images(
        input_path, 
        stacking_mode=stacking_method,
        kappa=kappa,
        max_iterations=iterations,
        batch_size=batch_size, 
        manual_reference_path=reference_path
    )
    
    input("\nAppuyez sur Entrée pour quitter...")