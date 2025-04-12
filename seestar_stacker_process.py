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

# ... [All previous functions remain unchanged except translated messages] ...

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
        update_progress("❌ No .fit/.fits files found")
        return None
    
    total_files = len(files)
    update_progress(f"\n🔍 Analyzing {total_files} images...")

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
    files_in_final_stack = 0

    # Try to load manual reference if provided
    if manual_reference_path:
        try:
            update_progress(f"\n📌 Loading manual reference image: {manual_reference_path}")
            fixed_reference_image = load_and_validate_fits(manual_reference_path)
            fixed_reference_header = fits.getheader(manual_reference_path)
            
            # Apply debayer to reference image if it's a raw image
            if fixed_reference_image.ndim == 2:
                fixed_reference_image = debayer_image(fixed_reference_image, bayer_pattern)
            elif fixed_reference_image.ndim == 3 and fixed_reference_image.shape[0] == 3:
                fixed_reference_image = np.moveaxis(fixed_reference_image, 0, -1)
                
            fixed_reference_image = cv2.normalize(fixed_reference_image, None, 0, 65535, cv2.NORM_MINMAX)
            update_progress(f"✅ Reference image loaded: dimensions: {fixed_reference_image.shape}")
        except Exception as e:
            update_progress(f"❌ Error loading manual reference image: {e}")
            fixed_reference_image = None

    # Find best reference image if not manually provided
    if fixed_reference_image is None:
        update_progress("\n⚙️ Finding best reference image...")
        sample_images = []
        sample_headers = []
        sample_files = []
        
        for f in tqdm(files[:min(batch_size*2, len(files))], desc="Analyzing images"):
            try:
                img_path = os.path.join(input_folder, f)
                img = load_and_validate_fits(img_path)
                hdr = fits.getheader(img_path)
                
                if np.std(img) > 5:
                    if img.ndim == 2:
                        img = debayer_image(img, bayer_pattern)
                    elif img.ndim == 3 and img.shape[0] == 3:
                        img = np.moveaxis(img, 0, -1)
                    
                    img = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX)
                    
                    sample_images.append(img)
                    sample_headers.append(hdr)
                    sample_files.append(f)
                else:
                    update_progress(f"⚠️ Image ignored (low variance): {f}")
            except Exception as e:
                update_progress(f"❌ Error analyzing {f}: {e}")
        
        if sample_images:
            medians = [np.median(img) for img in sample_images]
            ref_idx = np.argmax(medians)
            fixed_reference_image = sample_images[ref_idx]
            fixed_reference_header = sample_headers[ref_idx]
            update_progress(f"\n⭐ Reference used: {sample_files[ref_idx]}")
            
            ref_output_path = os.path.join(aligned_folder, "reference_image.fit")
            ref_data = np.moveaxis(fixed_reference_image, -1, 0).astype(np.uint16)
            
            new_header = fixed_reference_header.copy()
            new_header['NAXIS'] = 3
            new_header['NAXIS1'] = fixed_reference_image.shape[1]
            new_header['NAXIS2'] = fixed_reference_image.shape[0]
            new_header['NAXIS3'] = 3
            new_header['BITPIX'] = 16
            new_header.set('CTYPE3', 'RGB', 'RGB colors')
            
            fits.writeto(ref_output_path, ref_data, new_header, overwrite=True)
            update_progress(f"📁 Reference image saved: {ref_output_path}")
            
            sample_images = None
            sample_headers = None
            gc.collect()
        else:
            update_progress("❌ No valid reference image found")
            return None

    # Process images in batches
    for batch_start in range(0, len(files), batch_size):
        batch_files = files[batch_start:batch_start + batch_size]
        update_progress(f"\n🚀 Processing batch {batch_start // batch_size + 1} (images {batch_start + 1} to {batch_start + len(batch_files)})...")
        
        aligned_images = []
        aligned_headers = []
        aligned_paths = []
        unaligned_paths = []
        
        for i, f in enumerate(tqdm(batch_files, desc="Aligning")):
            try:
                img_path = os.path.join(input_folder, f)
                img = load_and_validate_fits(img_path)
                hdr = fits.getheader(img_path)
                
                if np.std(img) > 5:
                    if img.ndim == 2:
                        img = debayer_image(img, bayer_pattern)
                    elif img.ndim == 3 and img.shape[0] == 3:
                        img = np.moveaxis(img, 0, -1)
                    
                    img = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX)
                    
                    if img.ndim == 3:
                        aligned_channels = []
                        for c in range(3):
                            img_norm = cv2.normalize(img[:, :, c], None, 0, 1, cv2.NORM_MINMAX)
                            ref_norm = cv2.normalize(fixed_reference_image[:, :, c], None, 0, 1, cv2.NORM_MINMAX)
                            
                            aligned_channel, _ = aa.register(img_norm, ref_norm)
                            aligned_channels.append(aligned_channel)
                        
                        aligned_img = np.stack(aligned_channels, axis=-1)
                        aligned_img = cv2.normalize(aligned_img, None, 0, 65535, cv2.NORM_MINMAX)
                    else:
                        aligned_img, _ = aa.register(img, fixed_reference_image)
                        aligned_img = np.stack((aligned_img,) * 3, axis=-1)
                    
                    color_cube = np.moveaxis(aligned_img, -1, 0).astype(np.uint16)
                    
                    new_header = hdr.copy()
                    new_header['NAXIS'] = 3
                    new_header['NAXIS1'] = aligned_img.shape[1]
                    new_header['NAXIS2'] = aligned_img.shape[0]
                    new_header['NAXIS3'] = 3
                    new_header['BITPIX'] = 16
                    new_header.set('CTYPE3', 'RGB', 'RGB colors')
                    
                    out_path = os.path.join(aligned_folder, f"aligned_{batch_start + i:04}.fit")
                    fits.writeto(out_path, color_cube, new_header, overwrite=True)
                    
                    aligned_images.append(aligned_img)
                    aligned_headers.append(new_header)
                    aligned_paths.append(out_path)
                    
                    img = None
                    color_cube = None
                    
                    if "EXPTIME" in hdr:
                        total_exposure_time += hdr["EXPTIME"]
                else:
                    update_progress(f"Rejected (low quality): {f}")
            except Exception as e:
                update_progress(f"❌ Alignment failed for image {f}: {str(e)}")
                
                try:
                    original_path = os.path.join(input_folder, f)
                    out_path = os.path.join(unaligned_folder, f"unaligned_{f}")
                    with open(original_path, 'rb') as src, open(out_path, 'wb') as dst:
                        dst.write(src.read())
                    update_progress(f"⚠️ Unaligned image saved: {out_path}")
                    unaligned_paths.append(out_path)
                except Exception as copy_err:
                    update_progress(f"❌ Failed to copy original image: {copy_err}")
            
            processed_files += 1
            progress_percent = (processed_files / total_files) * 100
            remaining_time = estimate_remaining_time(processing_start_time, processed_files, total_files)
            update_progress(
                f"Processing {f} ({processed_files}/{total_files}) - "
                f"Estimated remaining time: {remaining_time}", 
                progress=progress_percent
            )
            
            gc.collect()
        
        if aligned_images:
            update_progress(f"⚙️ Creating sub-stack for batch {batch_start // batch_size + 1}...")
            batch_stack, batch_header = stack_batch(
                aligned_images, 
                aligned_headers,
                stacking_mode=stacking_mode,
                kappa=kappa,
                max_iterations=max_iterations
            )
            
            if batch_stack is not None:
                substack_path = os.path.join(substack_folder, f"sub_stack_{batch_start // batch_size + 1:03d}.fit")
                
                if batch_stack.ndim == 3 and batch_stack.shape[2] == 3:
                    batch_stack_fits = np.moveaxis(batch_stack, -1, 0).astype(np.float32)
                else:
                    batch_stack_fits = batch_stack.astype(np.float32)
                
                fits.writeto(substack_path, batch_stack_fits, batch_header, overwrite=True)
                update_progress(f"✅ Sub-stack saved: {substack_path}")
                
                batch_stacks.append(batch_stack)
                batch_stack_headers.append(batch_header)
                files_in_final_stack += len(aligned_images)
                
                batch_stack_fits = None
                
                update_progress("🧹 Deleting individual aligned files...")
                for path in aligned_paths:
                    try:
                        os.remove(path)
                    except Exception as e:
                        update_progress(f"⚠️ Failed to delete {path}: {e}")
                
                aligned_images = []
                aligned_headers = []
                aligned_paths = []
                gc.collect()
        
        if unaligned_paths:
            update_progress("🧹 Deleting unaligned files...")
            for path in unaligned_paths:
                try:
                    os.remove(path)
                except Exception as e:
                    update_progress(f"⚠️ Failed to delete {path}: {e}")
            
            unaligned_paths = []
        
        gc.collect()
    
    if not batch_stacks:
        update_progress("❌ No sub-stacks created, cannot generate final stack")
        return None
    
    update_progress(f"\n⭐ Creating final stack from {len(batch_stacks)} sub-stacks ({files_in_final_stack} images)...")
    
    final_stack, final_header = stack_batch(
        batch_stacks, 
        batch_stack_headers,
        stacking_mode=stacking_mode,
        kappa=kappa,
        max_iterations=max_iterations
    )
    
    batch_stacks = None
    batch_stack_headers = None
    gc.collect()
    
    if final_stack is None:
        update_progress("❌ Failed to create final stack")
        return None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_path = os.path.join(output_folder, f"FINAL_STACK_{stacking_mode}_{timestamp}.fit")
    
    if final_stack.ndim == 3 and final_stack.shape[2] == 3:
        final_stack_fits = np.moveaxis(final_stack, -1, 0).astype(np.float32)
    else:
        final_stack_fits = final_stack.astype(np.float32)
    
    final_header['NUMIMGS'] = files_in_final_stack, 'Total number of input images'
    final_header['EXPTIME'] = total_exposure_time, 'Total equivalent exposure time'
    final_header['DATE-STK'] = datetime.now().isoformat(), 'Date of stacking'
    
    fits.writeto(final_path, final_stack_fits, final_header, overwrite=True)
    
    final_stack = None
    final_stack_fits = None
    final_header = None
    gc.collect()
    
    total_time = time.time() - processing_start_time
    formatted_proc_time = format_exposure_time(total_time)
    
    update_progress(f"\n✅ Processing completed successfully!")
    update_progress(f"📁 Final stack: {final_path}")
    update_progress(f"📊 Files processed: {processed_files}")
    update_progress(f"📊 Files in final stack: {files_in_final_stack}")
    update_progress(f"⏱️ Total processing time: {formatted_proc_time}")
    if total_exposure_time > 0:
        formatted_exp_time = format_exposure_time(total_exposure_time)
        update_progress(f"⏱️ Total equivalent exposure time: {formatted_exp_time}")
    
    update_progress("\n🧹 Cleaning up sub-stacks...")
    for file in os.listdir(substack_folder):
        try:
            file_path = os.path.join(substack_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            update_progress(f"⚠️ Failed to delete {file}: {e}")
    
    try:
        if os.path.exists(unaligned_folder) and not os.listdir(unaligned_folder):
            os.rmdir(unaligned_folder)
            update_progress("🧹 'unaligned' folder deleted (empty)")
        
        if os.path.exists(aligned_folder) and (not os.listdir(aligned_folder) or (len(os.listdir(aligned_folder)) == 1 and os.path.exists(os.path.join(aligned_folder, "reference_image.fit")))):
            ref_path = os.path.join(aligned_folder, "reference_image.fit")
            if os.path.exists(ref_path):
                new_ref_path = os.path.join(output_folder, "reference_image.fit")
                shutil.move(ref_path, new_ref_path)
            
            if os.path.exists(aligned_folder):
                shutil.rmtree(aligned_folder)
                update_progress("🧹 'aligned_lights' folder deleted (empty)")
        
        if os.path.exists(substack_folder) and not os.listdir(substack_folder):
            os.rmdir(substack_folder)
            update_progress("🧹 'sub_stacks' folder deleted (empty)")
    except Exception as e:
        update_progress(f"⚠️ Folder cleanup error: {e}")
    
    return final_path

# ... [SeestarStacker class remains unchanged] ...

if __name__ == "__main__":
    input_path = input("📂 Enter the folder path containing FITS images: ").strip('"\' ')
    reference_path = input("📌 Enter reference image path (leave empty for auto-select): ").strip('"\' ')
    stacking_method = input("📋 Choose stacking method (mean, median, kappa-sigma, winsorized-sigma) [kappa-sigma]: ").strip() or "kappa-sigma"
    batch_size = int(input("🛠️ Enter batch size (e.g. 10): ") or 10)
    
    if stacking_method in ["kappa-sigma", "winsorized-sigma"]:
        kappa = float(input("📊 Enter kappa value (default: 2.5): ") or 2.5)
        iterations = int(input("🔄 Enter number of iterations (default: 3): ") or 3)
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
    
    input("\nPress Enter to exit...")