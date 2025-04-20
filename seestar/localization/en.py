"""
Fichier de traductions anglaises pour Seestar.
"""

EN_TRANSLATIONS = {
    # Main interface
    'title': "Seestar Stacker",
    'input_folder': "Input folder:",
    'output_folder': "Output folder:",
    'browse': "Browse", # Generic key
    'browse_input_button': "Browse...", # Unique key for button
    'browse_output_button': "Browse...", # Unique key for button
    'browse_ref_button': "Browse...", # Unique key for button
    'Folders': "Folders",

    # Options
    'options': "Options",
    'stacking_method': "Method:",
    'kappa_value': "Kappa:",
    'batch_size': "Batch size:",
    # 'remove_aligned': "Remove aligned files", # Setting removed
    'apply_denoise': "Apply Denoise",

    # Stacking methods
    'mean': "mean", 'median': "median", 'kappa-sigma': "kappa-sigma", 'winsorized-sigma': "winsorized-sigma",

    # Alignment
    'reference_image': "Reference (optional):",
    'alignment_start': "⚙️ Starting image alignment...",
    'using_aligned_folder': "✅ Using aligned folder: {}",
    'Alignment & Hot Pixels': 'Alignment & Hot Pixels',
    'Getting reference image...': "⭐ Getting reference image...",
    'Failed to get reference image.': "❌ Failed to get reference image.",
    'Reference image ready': "✅ Reference image ready",
    'Aligning Batch': "📐 Aligning Batch",
    'aligned': "aligned",
    'alignment failed': "alignment failed",
    'Error in alignment worker': "❗️ Error in alignment worker",
    'Auto batch size': "🧠 Auto batch size",

    # Progress
    'progress': "Progress",
    'estimated_time': "ETA:",
    'elapsed_time': "Elapsed:",
    'calculating': "Calculating...",
    'Remaining:': "Remaining:", # Key for static label
    'Additional:': "Additional:", # Key for static label

    # Control buttons
    'start': "Start",
    'stop': "Stop",
    'add_folder_button': "Add Folder",
    'reset_zoom_button': "Reset Zoom",

    # Messages & Status
    'error': "Error", 'warning': "Warning", 'info': "Information", 'quit': "Quit",
    'select_folders': "Please select input and output folders.",
    'input_folder_invalid': "Invalid input folder",
    'output_folder_invalid': "Invalid output folder/cannot create",
    'no_fits_found': "No .fit/.fits files found in input folder.",
    'stacking_start': "⚙️ Starting image processing...",
    'stacking_stopping': "⚠️ Stopping...",
    'stacking_finished': "🏁 Processing Finished",
    'stacking_error_msg': "Processing Error:",
    'stacking_complete_msg': "Processing complete! Final stack:",
    'stop_requested': "⚠️ Stop requested, please wait...",
    'processing_stopped': "🛑 Processing stopped by user.",
    'processing_stopped_additional': "🛑 Processing stopped during additional folders.",
    # 'processing_complete': "✅ Processing completed successfully!", # Maybe redundant now
    'no_stacks_created': "⚠️ No stacks were created.",
    'stacks_created': "✅ Stacks created.",
    'preview': "Preview",
    'no_current_stack': "No Active Stack",
    'image_info_waiting': "Image info: waiting...",
    'stretch_preview': "Stretch Preview",
    'no_files_waiting': "No files waiting",
    'no_additional_folders': "None",
    # 'Main processing failed': "Main processing failed", # Less specific now
    # 'Processing ended prematurely': "Processing ended prematurely", # Less specific now
    'Error during cleanup': "Error during cleanup",
    'Output folder created': "Output folder created",
    'Error reading input folder': "Error reading input folder",
    'Start processing to add folders': "Processing must be started to add additional folders.",
    'Select Additional Images Folder': "Select Folder with Additional Images",
    'Folder not found': "Folder not found",
    'Input folder cannot be added': "The main input folder cannot be added as an additional folder.",
    'Folder already added': "This folder is already in the additional list.",
    'Folder contains no FITS': "The selected folder does not contain any FITS files (.fit or .fits).",
    'Error reading folder': "Error reading folder",
    'Folder added': "Folder added",
    'files': "files",
    '1 additional folder': "1 additional folder",
    '{count} additional folders': "{count} additional folders",
    'Select Input Folder': "Select Input Folder",
    'Select Output Folder': "Select Output Folder",
    'Select Reference Image (Optional)': "Select Reference Image (Optional)",
    'quit_while_processing': "Processing is active. Quit anyway?",
    'Stack': "Stack", 'imgs': "imgs", 'Object': "Object", 'Date': "Date", 'Exposure (s)': "Exp (s)", 'Gain': "Gain",
    'Offset': "Offset", 'Temp (°C)': "Temp (°C)", 'Images': "Images", 'Method': "Method", 'Filter': "Filter", 'Bayer': "Bayer",
    'No image info available': "No image info available",
    # 'Main processing finished': "Main processing finished", # Less specific now
    # 'images in final stack': "images in final stack", # Less specific now
    # 'No batches were stacked': "No batches were stacked", # Less specific now
    # 'Processing': "Processing", 'additional folders': "additional folders", # Covered elsewhere
    'Reference image not found for additional folders': "Reference image not found for additional folders",
    'Using reference': "Using reference", 'Processing additional folder': "Processing additional folder",
    'No FITS files in': "No FITS files in", 'Skipped': "Skipped", 'images found': "images found",
    # 'Batch size': "Batch size", # Covered elsewhere
    'Folder': "Folder", 'Batch': "Batch", 'processed': "processed", 'images added': "images added",
    'Error processing folder': "Error processing folder",
    'Applying denoising to final stack': "Applying denoising to final stack",
    'Final denoised stack saved': "Final denoised stack saved",
    'Saving final stack with metadata': "Saving final stack with metadata",
    'Final stack with metadata saved': "Final stack with metadata saved",
    'Additional folder processing complete': "Additional folder processing complete",
    'Additional folder processing finished with errors': "Additional folder processing finished with errors",
    # 'No files provided for stacking': "No files provided for stacking", # Less specific
    'Error loading/validating': "Error loading/validating",
    'No valid images to stack after loading': "No valid images to stack after loading",
    'Unknown stacking method': "Unknown stacking method",
    'using \'mean\'': "using 'mean'",
    'Stacking failed (result is None)': "Stacking failed (result is None)",
    'Dimension mismatch during combine': "Dimension mismatch during combine",
    'Skipping combination': "Skipping combination",
    'Error combining stacks': "Error combining stacks",
    'Cleaning temporary files': "Cleaning temporary files",
    'Cannot delete': "Cannot delete",
    'Error scanning folder': "Error scanning folder",
    'temporary files removed': "temporary files removed",
    'Error reading metadata from': "Error reading metadata from",
    'Failed to save final metadata stack': "Failed to save final metadata stack",
    'Error saving final FITS stack': "Error saving final FITS stack",
    'Denoising failed': "Denoising failed",
    'Error displaying reference image': "Error displaying reference image",
    'Error saving cumulative stack': "Error saving cumulative stack",
    'Could not find reference for final metadata stack': "Could not find reference for final metadata stack",
    'Error preparing final metadata stack': "Error preparing final metadata stack",
    'Welcome!': "Welcome!",
    'Select input/output folders.': "Select input/output folders.",
    'Preview:': "Preview:",
    'No Image Data': "No Image Data", 'Preview Error': "Preview Error", 'Preview Update Error': "Preview Update Error",
    'No FITS files in input': "No FITS files in input",
    # 'Alignment failed for batch': "Alignment failed for batch", # Less specific now
    # 'Main processing OK, but errors in additional folders': "Main processing OK, but errors in additional folders", # Less specific
    # 'Critical error in main processing': "Critical error in main processing", # Less specific
    'Error loading preview image': "Error loading preview image",
    'Error loading preview (invalid format)': "Error loading preview (invalid format)",
    'Error during debayering': "Error during debayering",
    'Invalid or missing BAYERPAT': "Invalid or missing BAYERPAT",
    'Treating as grayscale': "Treating as grayscale",
    'Error loading final stack preview': "Error loading final stack preview",

    # Hot pixels
    'hot_pixels_correction': 'Hot Pixel Correction',
    'perform_hot_pixels_correction': 'Correct hot pixels',
    'hot_pixel_threshold': 'Threshold:',
    'neighborhood_size': 'Neighborhood:',
    'neighborhood_size_adjusted': "Neighborhood size adjusted to {size} (must be odd)",

    # Processing mode
    'Final Stack': 'Final Stack', # LabelFrame key

    # Aligned files counter
    'aligned_files_label': "Aligned:", # Static label text (placeholder if StringVar is used directly)
    'aligned_files_label_format': "Aligned: {count}", # Format string for display

    # Total Exposure Time <-- ADDED Key
    'Total Exp (s)': "Total Exp (s)",
}