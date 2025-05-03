from astropy.io import fits
import numpy as np # Import numpy too

# --- IMPORTANT: REPLACE THIS WITH THE ACTUAL PATH ---
file_path = 'E:/astro photos/seestar s 50/M 94/out/stack_cumulative.fit'
# ----------------------------------------------------

try:
    print(f"\n--- FITS Info for: {file_path} ---")
    fits.info(file_path) # Prints HDU structure summary

    with fits.open(file_path) as hdul:
        print("\n--- Primary HDU (HDU 0) Header ---")
        # Using repr() gives more detail than just print()
        print(repr(hdul[0].header))

        # Let's check the data type and shape in the first few HDUs
        for i, hdu in enumerate(hdul):
             print(f"\n--- Details for HDU {i} ---")
             print(f"Name: {hdu.name}") # HDU name, if any
             if hdu.data is not None:
                 print(f"Data Shape: {hdu.data.shape}")
                 print(f"Data Type (dtype): {hdu.data.dtype}")
                 # Try to get min/max safely, ignoring non-finite values
                 try:
                     finite_data = hdu.data[np.isfinite(hdu.data)]
                     if finite_data.size > 0:
                         print(f"Data Range (finite values): [{np.min(finite_data):.4g}, {np.max(finite_data):.4g}]")
                     else:
                         print("Data contains only non-finite values (NaN/Inf) or is empty.")
                 except Exception as stat_err:
                     print(f"Could not calculate data range: {stat_err}")

             else:
                 print("Data: None")

except FileNotFoundError:
    print(f"\n*** ERROR: File not found at: {file_path} ***")
except Exception as e:
    print(f"\n*** ERROR opening or reading FITS file: {e} ***")
    import traceback
    traceback.print_exc() # Print full traceback for other errors