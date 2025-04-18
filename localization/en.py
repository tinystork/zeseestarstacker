"""
Fichier de traductions anglaises pour Seestar.
"""

EN_TRANSLATIONS = {
    # Main interface
    'title': "Seestar Stacker",
    'input_folder': "Input folder:",
    'output_folder': "Output folder:",
    'browse': "Browse",
    
    # Options
    'options': "Options",
    'stacking_method': "Stacking method:",
    'kappa_value': "Kappa value:",
    'batch_size': "Batch size (0 for auto):",
    'remove_aligned': "Remove aligned images after stacking",
    'apply_denoise': "Apply denoising to final stack",
    
    # Stacking methods
    'mean': "mean",
    'median': "median",
    'kappa-sigma': "kappa-sigma",
    'winsorized-sigma': "winsorized-sigma",
    
    # Processing options
    'processing_options': "Processing Options",
    'progressive_stacking_start': "⚙️ Starting progressive stacking...",
    
    # Alignment
    'alignment': "Alignment",
    'perform_alignment': "Perform alignment before stacking",
    'reference_image': "Reference image (leave empty for automatic selection):",
    'alignment_start': "⚙️ Starting image alignment...",
    'using_aligned_folder': "✅ Using aligned folder: {}",
    
    # Progress
    'progress': "Progress",
    'estimated_time': "Estimated time remaining:",
    'elapsed_time': "Elapsed time:",
    'calculating': "Calculating...",
    
    # Control buttons
    'start': "Start",
    'stop': "Stop",
    
    # Messages
    'error': "Error",
    'select_folders': "Please select input and output folders.",
    'stacking_start': "⚙️ Starting image stacking...",
    'stop_requested': "⚠️ Stop requested, please wait...",
    'processing_stopped': "Processing stopped by user",
    'processing_completed': "Processing completed successfully!",
    'no_stacks_created': "No stacks were created",
    'stacks_created': "stacks were created",
    'neighborhood_size_adjusted': "Neighborhood size was adjusted to",
    
    # Hot pixels
    'hot_pixels_correction': 'Hot Pixels Correction',
    'perform_hot_pixels_correction': 'Perform hot pixels correction',
    'hot_pixel_threshold': 'Detection threshold:',
    'neighborhood_size': 'Neighborhood size:',
    
    # Incremental stacking
    'incremental_stacking': 'Incremental Stacking',
    'use_incremental_stacking': 'Use incremental stacking (saves disk space)',
    'keep_aligned_images': 'Keep aligned images',
    'keep_intermediate_stacks': 'Keep intermediate stacks',
    'incremental_stacking_start': '⚙️ Starting incremental stacking...',
    'real_time_preview': 'Real-time Preview',
    'enable_preview': 'Enable real-time preview',
    'queue_mode': 'Queue Mode',
    'use_queue_mode': 'Use queue mode',
    'use_traditional_mode': 'Use traditional stacking',
    'remove_processed': 'Remove processed images',
    'queue_start': '⚙️ Starting queue processing...',
    'queue_completed': 'Queue processing completed.',
}