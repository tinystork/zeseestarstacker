# --- START OF FILE seestar/localization/en.py ---
"""
English translation file for Seestar Stacker.
"""

EN_TRANSLATIONS = {
    # Main interface & Common
    'title': "Seestar Stacker",
    'error': "Error", 'warning': "Warning", 'info': "Information", 'quit': "Quit",
    'browse': "Browse", # Generic key
    'browse_input_button': "Browse...", # Unique key for button
    'browse_output_button': "Browse...", # Unique key for button
    'browse_ref_button': "Browse...", # Unique key for button
    'drizzle_options_frame_label': "Drizzle Options",
    'drizzle_activate_check': "Enable Drizzle (experimental, slow)",
    'drizzle_scale_label': "Scale Factor:",
    'drizzle_radio_2x_label': "x2",
    'drizzle_radio_3x_label': "x3",
    'drizzle_radio_4x_label': "x4",

    # --- Control Tabs ---
    'tab_stacking': "Stacking",
    'tab_preview': "Preview",

    # --- Stacking Tab ---
    'Folders': "Folders",
    'input_folder': "Input:",
    'output_folder': "Output:",
    'reference_image': "Reference (Opt.):",
    'options': "Stacking Options",
    'stacking_method': "Method:",
    'kappa_value': "Kappa:",
    'batch_size': "Batch Size:",
    'batch_size_auto': "(0=auto)", # Gardé même si 0 n'est plus auto pour compatibilité affichage
    'hot_pixels_correction': 'Hot Pixel Correction',
    'perform_hot_pixels_correction': 'Correct hot pixels',
    'hot_pixel_threshold': 'Threshold:',
    'neighborhood_size': 'Neighborhood:',
    'post_proc_opts_frame_label': "Post-Processing Options",
    'cleanup_temp_check_label': "Cleanup temporary files after processing",
    'quality_weighting_frame': "Quality Weighting",
    'enable_weighting_check': "Enable weighting",
    'weighting_metrics_label': "Metrics:",
    'weight_snr_check': "SNR",
    'weight_stars_check': "Star Count",
    'snr_exponent_label': "SNR Exp.:",
    'stars_exponent_label': "Stars Exp.:",
    'min_weight_label': "Min Weight:",

    # --- Preview Tab ---
    'white_balance': "White Balance (Preview)",
    'wb_r': "R Gain:",
    'wb_g': "G Gain:",
    'wb_b': "B Gain:",
    'auto_wb': "Auto WB",
    'reset_wb': "Reset WB",
    'stretch_options': "Stretch (Preview)",
    'stretch_method': "Method:",
    'stretch_bp': "Black:",
    'stretch_wp': "White:",
    'stretch_gamma': "Gamma:",
    'auto_stretch': "Auto Stretch",
    'reset_stretch': "Reset Stretch",
    'image_adjustments': "Image Adjustments",
    'brightness': "Brightness:",
    'contrast': "Contrast:",
    'saturation': "Saturation:",
    'reset_bcs': "Reset Adjust.",

    # --- Progress Area ---
    'progress': "Progress",
    'estimated_time': "ETA:",
    'elapsed_time': "Elapsed:",
    'Remaining:': "Remaining:", # Key for static label
    'Additional:': "Additional:", # Key for static label
    'aligned_files_label': "Aligned:", # Static label text (placeholder)
    'aligned_files_label_format': "Aligned: {count}", # Format string for display
    'global_eta_label': "Global ETA:", # Global ETA label

    # --- Preview Area (Right Panel) ---
    'preview': "Preview",
    'histogram': "Histogram",

    # --- Control Buttons ---
    'start': "Start",
    'stop': "Stop",
    'add_folder_button': "Add Folder",
    # NOUVEAU: Clés pour les nouveaux boutons (utiliser le nom de variable comme référence)
    'copy_log_button_text': "Copy",
    'open_output_button_text': "Open Output",
    'show_folders_button_text': "View Inputs",
    'Select Input Folder': "Select Input Folder",

    'input_folders_title': "Input Folder List",
    'no_input_folder_set': "No input folder has been selected yet.",

    # --- Dialog Titles ---
    'Select Input Folder': "Select Input Folder",
    'Select Output Folder': "Select Output Folder",
    'Select Reference Image (Optional)': "Select Reference Image (Optional)",
    'Select Additional Images Folder': "Select Folder with Additional Images",

    # --- Status Messages & Errors ---
    'select_folders': "Please select input and output folders.",
    'input_folder_invalid': "Invalid input folder",
    'output_folder_invalid': "Invalid output folder/cannot create",
    'Output folder created': "Output folder created",
    'no_fits_found': "No .fit/.fits files found in input folder. Start anyway?",
    'Error reading input folder': "Error reading input folder",
    'stacking_start': "⚙️ Starting processing...",
    'stacking_stopping': "⚠️ Stopping...",
    'stacking_finished': "🏁 Processing Finished",
    'stacking_error_msg': "Processing Error:",
    'stacking_complete_msg': "Processing complete! Final stack:",
    'stop_requested': "⚠️ Stop requested, finishing current step...",
    'processing_stopped': "🛑 Processing stopped by user.",
    'no_stacks_created': "⚠️ No stacks were created.",
    'Failed to start processing.': "Failed to start processing.",
    'image_info_waiting': "Image info: waiting...",
    'no_files_waiting': "No files waiting",
    'no_additional_folders': "None",
    '1 additional folder': "1 add. folder",
    '{count} additional folders': "{count} add. folders",
    'Start processing to add folders': "Processing must be started to add additional folders.", # Reste pertinent pour l'ajout pendant le process
    'Processing not active or finished.': 'Processing not active or finished.',
    'Folder not found': "Folder not found",
    'Input folder cannot be added': "The main input folder cannot be added.",
    'Output folder cannot be added': "The output folder cannot be added.",
    'Cannot add subfolder of output folder': "Cannot add a subfolder of the output folder.",
    'Folder already added': "This folder is already in the list.",
    'Folder contains no FITS': "Folder contains no FITS files.",
    'Error reading folder': "Error reading folder",
    'Folder added': "Folder added",
    'Folder not added (already present, empty, or error)': "Folder not added (already present, empty, or error)",
    'quit_while_processing': "Processing is active. Quit anyway?",
    'Error during debayering': "Error during debayering",
    'Invalid or missing BAYERPAT': "Invalid or missing BAYERPAT",
    'Treating as grayscale': "Treating as grayscale",
    'Error loading preview image': "Error loading preview image",
    'Error loading preview (invalid format?)': "Error loading preview (invalid format?)",
    'Error loading final stack preview': "Error loading final stack preview",
    'Error loading final preview': "Error loading final preview",
    'No Image Data': "No Image Data",
    'Preview Error': "Preview Error",
    'Preview Processing Error': "Preview Processing Error",
    'Welcome!': "Welcome!",
    'Select input/output folders.': "Select input/output folders.",
    'Auto WB requires a color image preview.': "Auto WB requires a color image preview.",
    'Error during Auto WB': 'Error during Auto WB',
    'Auto Stretch requires an image preview.': "Auto Stretch requires an image preview.",
    'Error during Auto Stretch': 'Error during Auto Stretch',
    'Total Exp (s)': "Total Exp (s)",
    'processing_report_title': "Processing Summary",
    'report_images_stacked': "Images Stacked:",
    'report_total_exposure': "Total Exposure:",
    'report_total_time': "Total Processing Time:",
    'report_seconds': "seconds",
    'report_minutes': "minutes",
    'report_hours': "hours",
    'eta_calculating': "Calculating...",
    # --- Weighting Info Display ---
    'Weighting': 'Weighting', # Key for label 'WGHT_ON' in header
    'W. Metrics': 'W. Metrics', # Key for label 'WGHT_MET' in header
    'weighting_enabled': "Enabled", # Value for WGHT_ON=True
    'weighting_disabled': "Disabled", # Value for WGHT_ON=False
    'drizzle_wht_threshold_label': "WHT Threshold%:",

    # --- NEW: Texts for Drizzle Warning ---
    'drizzle_warning_title': "Drizzle Warning",
    'drizzle_warning_text': (
    "Drizzle processing is enabled.\n\n",
    "- This is experimental and may be slow.\n",
    "- It will create temporary files that can consume significant disk space (potentially similar to the input image size).\n",
    "- The live preview will show a standard stack; Drizzle will be applied at the very end.\n\n",
    "Continue with Drizzle?"
    ),
    # --- End New Texts ---

}
# --- END OF FILE seestar/localization/en.py ---