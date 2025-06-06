"""
Fonctions utilitaires pour le traitement d'images astronomiques.
"""
import numpy as np
import cv2
import os # Added for exists check
import traceback # Added for better error reporting
from .image_processing import load_and_validate_fits # Keep relative import

# Try importing psutil, but make it optional
try:
    import psutil
    _psutil_available = True
except ImportError:
    _psutil_available = False
    print("Optional dependency 'psutil' not found. Automatic batch size estimation may be limited.")
# --- Add a global check for CUDA availability ONCE ---
_cuda_available = False
_cuda_checked = False
# --- Global check for CuPy and CUDA availability ---
_cupy_available = False
_cupy_checked = False

def check_cuda():
    """Checks if OpenCV reports CUDA devices and sets a global flag."""
    global _cuda_available, _cuda_checked
    if _cuda_checked:
        return _cuda_available
    try:
        # Make sure opencv-contrib-python is potentially installed
        if not hasattr(cv2, 'cuda'):
             print("DEBUG: cv2.cuda module not found (likely opencv-python, not opencv-contrib-python or CUDA not supported in build).")
             _cuda_available = False
        elif cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("DEBUG: CUDA device(s) detected by OpenCV.")
            cv2.cuda.printCudaDeviceInfo(cv2.cuda.getDevice()) # Print info about the default device
            _cuda_available = True
        else:
            print("DEBUG: No CUDA devices detected by OpenCV.")
            _cuda_available = False
    except Exception as e:
        print(f"DEBUG: Error checking for CUDA devices: {e}")
        _cuda_available = False
    finally:
        _cuda_checked = True
    return _cuda_available

def check_cupy_cuda():
    """Checks if CuPy is installed and can access a CUDA device."""
    global _cupy_available, _cupy_checked
    if _cupy_checked:
        return _cupy_available

    try:
        import cupy
        # Check if CuPy can detect a CUDA device
        if cupy.cuda.is_available():
            device_id = cupy.cuda.runtime.getDevice()
            device_props = cupy.cuda.runtime.getDeviceProperties(device_id)
            print(f"DEBUG: CuPy detected CUDA Device {device_id}: {device_props['name'].decode()}")
            # Optional: Check compute capability if needed
            # major = device_props['major']
            # minor = device_props['minor']
            # if major < 3: # Example: Require compute capability 3.0+
            #     print(f"Warning: CuPy detected GPU compute capability {major}.{minor}, which might be too low.")
            #     _cupy_available = False
            # else:
            _cupy_available = True
        else:
            print("DEBUG: CuPy imported, but no CUDA device is available/detected by CuPy.")
            _cupy_available = False
    except ImportError:
        print("DEBUG: CuPy library not found. Stacking will use CPU (NumPy).")
        _cupy_available = False
    except cupy.cuda.runtime.CUDARuntimeError as e:
         print(f"DEBUG: CuPy CUDA runtime error during check: {e}. Falling back to CPU.")
         _cupy_available = False
    except Exception as e:
        print(f"DEBUG: Unexpected error during CuPy check: {e}")
        _cupy_available = False
    finally:
        _cupy_checked = True

    return _cupy_available

def apply_denoise(image, strength=1):
    """
    Applies Non-Local Means denoising using OpenCV. Uses CUDA if available,
    otherwise falls back to CPU.
    Input image is expected to be float32 (0-1 range).

    Parameters:
        image (numpy.ndarray): Image to denoise (HxW or HxWx3, float32, 0-1).
        strength (int): Denoising strength ('h' parameter). Recommended: 3 to 15.

    Returns:
        numpy.ndarray: Denoised image (float32, 0-1).
    """
    if image is None:
        print("Warning: apply_denoise received a None image.")
        return None

    # Check CUDA availability (only performs the check once)
    use_cuda = check_cuda()

    # Convert image float (0-1) to uint8 (0-255) for OpenCV
    image_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)

    denoised_uint8 = None
    denoised_bgr_uint8 = None
    h_param = float(strength) # Parameter 'h' in OpenCV functions

    try:
        # --- Grayscale Image ---
        if image_uint8.ndim == 2:
            if use_cuda:
                try:
                    # print(f"DEBUG: Attempting CUDA denoising (grayscale) h={h_param}")
                    # Upload image data to GPU memory
                    gpu_frame = cv2.cuda_GpuMat()
                    gpu_frame.upload(image_uint8)

                    # Create denoising object and apply
                    # Note: CUDA version might have slightly different parameter names/needs
                    # Adjust templateWindowSize and searchWindowSize if needed
                    dn = cv2.cuda.createFastNlMeansDenoising()
                    gpu_denoised = dn.apply(gpu_frame, h_param)  # GPU processing

                    # Download result back from GPU memory to CPU memory
                    denoised_uint8 = gpu_denoised.download()
                    # print("DEBUG: CUDA grayscale denoising successful.")
                except cv2.error as cuda_err:
                    print(f"Warning: CUDA grayscale denoising failed: {cuda_err}. Falling back to CPU.")
                    use_cuda = False # Fallback for this call if CUDA fails
                except Exception as e: # Catch other potential CUDA errors
                    print(f"Warning: Unexpected error during CUDA grayscale denoising: {e}. Falling back to CPU.")
                    traceback.print_exc(limit=1)
                    use_cuda = False # Fallback

            # Fallback to CPU if CUDA not available or failed
            if not use_cuda or denoised_uint8 is None:
                # print(f"DEBUG: Using CPU denoising (grayscale) h={h_param}")
                denoised_uint8 = cv2.fastNlMeansDenoising(
                    image_uint8, None, h=h_param, templateWindowSize=7, searchWindowSize=21)

        # --- Color Image ---
        elif image_uint8.ndim == 3 and image_uint8.shape[-1] == 3:
             # OpenCV color functions expect BGR format
             image_bgr_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)

             if use_cuda:
                 try:
                     # print(f"DEBUG: Attempting CUDA denoising (color) h={h_param}, hColor={h_param}")
                     gpu_frame = cv2.cuda_GpuMat()
                     gpu_frame.upload(image_bgr_uint8)

                     # Create color denoising object and apply
                     # Note: CUDA version might handle hColor differently or automatically
                     dn = cv2.cuda.createFastNlMeansDenoisingColored()
                     gpu_denoised = dn.apply(gpu_frame, h_param, h_param)

                     denoised_bgr_uint8 = gpu_denoised.download()
                     # print("DEBUG: CUDA color denoising successful.")
                 except cv2.error as cuda_err:
                    print(f"Warning: CUDA color denoising failed: {cuda_err}. Falling back to CPU.")
                    use_cuda = False # Fallback
                 except Exception as e:
                    print(f"Warning: Unexpected error during CUDA color denoising: {e}. Falling back to CPU.")
                    traceback.print_exc(limit=1)
                    use_cuda = False # Fallback

             # Fallback to CPU if CUDA not available or failed
             if not use_cuda or denoised_bgr_uint8 is None:
                 # print(f"DEBUG: Using CPU denoising (color) h={h_param}, hColor={h_param}")
                 denoised_bgr_uint8 = cv2.fastNlMeansDenoisingColored(
                     image_bgr_uint8, None, h=h_param, hColor=h_param,
                     templateWindowSize=7, searchWindowSize=21)

             # Convert result back to RGB
             denoised_uint8 = cv2.cvtColor(denoised_bgr_uint8, cv2.COLOR_BGR2RGB)

        # --- Invalid Input Shape ---
        else:
            print(f"Warning: apply_denoise received image with unexpected shape {image.shape}. Returning original.")
            return image # Return original float32 image if shape is wrong

        # Convert denoised uint8 back to float32 (0-1 range)
        denoised_float = denoised_uint8.astype(np.float32) / 255.0
        return denoised_float

    except cv2.error as cv_err: # Catch general OpenCV errors
         print(f"OpenCV Error during denoising setup/conversion: {cv_err}")
         traceback.print_exc()
         return image # Return original float32 image
    except Exception as e:
        print(f"Unexpected error during denoising: {e}")
        traceback.print_exc()
        return image # Return original float32 image



def estimate_batch_size(sample_image_path=None, available_memory_percentage=70):
    """
    Estime la taille de lot optimale en fonction de la mémoire disponible.
    CORRIGÉ: Gère correctement le tuple retourné par load_and_validate_fits.

    Parameters:
        sample_image_path: Chemin vers une image exemple pour estimer la taille mémoire
        available_memory_percentage: Pourcentage de la mémoire disponible à utiliser (0-100)

    Returns:
        int: Taille de lot estimée, au moins 3 et au plus 50
    """
    # Default batch size if estimation fails
    default_batch_size = 10

    if not _psutil_available:
        print("psutil not available, using default batch size:", default_batch_size)
        return default_batch_size

    try:
        # Obtenir la mémoire disponible (en octets)
        mem = psutil.virtual_memory()
        available_memory = mem.available

        # N'utiliser qu'un pourcentage de la mémoire disponible
        usable_memory = available_memory * (available_memory_percentage / 100.0)

        # Estimer la taille d'une image en mémoire pendant le traitement
        single_image_size_bytes = 0
        if sample_image_path and os.path.exists(sample_image_path):
            img_data_for_estimation = None # Initialiser
            try:
                # Load image to get dimensions and type (returns float32 0-1)
                loaded_tuple = load_and_validate_fits(sample_image_path) # APPEL MODIFIÉ

                # --- DÉBUT DE LA CORRECTION ---
                if loaded_tuple and loaded_tuple[0] is not None:
                    img_data_for_estimation = loaded_tuple[0] # Déballer l'array image
                else:
                    # Si load_and_validate_fits retourne None ou si les données sont None,
                    # img_data_for_estimation restera None.
                    # Le ValueError sera levé plus bas si img_data_for_estimation est None.
                    pass # img_data_for_estimation est déjà None
                # --- FIN DE LA CORRECTION ---

                if img_data_for_estimation is None: # Vérifier après la tentative de déballage
                    raise ValueError(f"Failed to load sample image: {sample_image_path}")

                # Estimate memory usage during processing (alignment + stacking buffer)
                memory_factor = 6
                h, w = img_data_for_estimation.shape[:2] # Utiliser img_data_for_estimation
                channels_out = 3 # Assume color output for worst-case size
                bytes_per_float = 4
                single_image_size_bytes = h * w * channels_out * bytes_per_float * memory_factor

            except Exception as img_e:
                 print(f"Warning: Could not load/analyze sample image {sample_image_path} for size estimation: {img_e}")
                 single_image_size_bytes = 0 # Fallback
        else:
            print("Warning: No valid sample image path provided for size estimation.")
            single_image_size_bytes = 0 # Fallback


        # Fallback estimation if image loading failed or no path provided
        if single_image_size_bytes <= 0:
            print("Using fallback image size estimation (approx. 4MP color image).")
            single_image_size_bytes = 2000 * 2000 * 3 * 4 * 6


        # Safety factor for other system usage and Python overhead
        safety_factor = 1.5

        if single_image_size_bytes <= 0:
             print("Error: Calculated image size is zero or negative.")
             return default_batch_size

        estimated_batch = int(usable_memory / (single_image_size_bytes * safety_factor))
        estimated_batch = max(3, min(50, estimated_batch)) # Limites raisonnables

        print(f"Mémoire disponible: {available_memory / (1024**3):.2f} Go "
              f"(Utilisable: {usable_memory / (1024**3):.2f} Go)")
        print(f"Taille estimée par image (avec overhead): {single_image_size_bytes / (1024**2):.2f} Mo")
        print(f"Taille de lot estimée: {estimated_batch}")

        return estimated_batch

    except Exception as e:
        print(f"Erreur lors de l'estimation de la taille de lot: {e}")
        traceback.print_exc()
        print(f"Utilisation de la taille de lot par défaut : {default_batch_size}")
        return default_batch_size


