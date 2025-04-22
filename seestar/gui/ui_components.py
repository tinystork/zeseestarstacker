"""
UI components module for Seestar.
Manages the creation and update of all interface elements.
"""

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk


class UIComponentsManager:
	"""
	Manages the creation and update of UI components for Seestar.
	"""
	
	def __init__(self, main_window):
		"""
		Initialize the UI components manager.
		
		Args:
			main_window: The main SeestarStackerGUI instance
		"""
		self.main = main_window
		
		# References to UI elements that need to be accessed from other classes
		self.preview_canvas = None
		self.current_stack_label = None
		self.image_info_text = None
		self.progress_bar = None
		self.status_text = None
		self.start_button = None
		self.stop_button = None
		self.add_files_button = None
		
		# Labels that need to be updated when language changes
		self.input_label = None
		self.output_label = None
		self.browse_input_button = None
		self.browse_output_button = None
		self.browse_ref_button = None
		self.reference_label = None
		self.stacking_method_label = None
		self.kappa_label = None
		self.batch_size_label = None
		self.remaining_time_label = None
		self.elapsed_time_label = None
		self.hot_pixels_frame = None
		self.hot_pixels_check = None
		self.hot_pixel_threshold_label = None
		self.neighborhood_size_label = None
		self.alignment_frame = None
		self.queue_check = None
		self.traditional_check = None
		self.remove_processed_check = None
	
	def create_layout(self):
		"""Create layout of interface widgets."""
		# Create a main frame that will contain everything
		main_frame = ttk.Frame(self.main.root)
		main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

		# Create a left frame for controls
		left_frame = ttk.Frame(main_frame)
		left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

		# Create a right frame for preview
		right_frame = ttk.Frame(main_frame)
		right_frame.pack(side=tk.RIGHT, fill=tk.BOTH,
						 expand=True, padx=(10, 0))

		# -----------------------
		# LEFT AREA (CONTROLS)
		# -----------------------

		# Language selection
		language_frame = ttk.Frame(left_frame)
		language_frame.pack(fill=tk.X, pady=5)
		ttk.Label(language_frame, text="Language / Langue:").pack(side=tk.LEFT)
		language_combo = ttk.Combobox(
			language_frame, textvariable=self.main.language_var, width=15)
		language_combo['values'] = ('English', 'Français')
		language_combo.pack(side=tk.LEFT, padx=5)
		language_combo.bind('<<ComboboxSelected>>', self.main.change_language)

		# Input folder
		input_frame = ttk.Frame(left_frame)
		input_frame.pack(fill=tk.X, pady=5)
		self.input_label = ttk.Label(input_frame, text=self.main.tr('input_folder'))
		self.input_label.pack(side=tk.LEFT)
		ttk.Entry(input_frame, textvariable=self.main.input_path, width=40).pack(
			side=tk.LEFT, padx=5, fill=tk.X, expand=True)
		self.browse_input_button = ttk.Button(
			input_frame, text=self.main.tr('browse'), command=self.main.browse_input)
		self.browse_input_button.pack(side=tk.RIGHT)

		# Output folder
		output_frame = ttk.Frame(left_frame)
		output_frame.pack(fill=tk.X, pady=5)
		self.output_label = ttk.Label(
			output_frame, text=self.main.tr('output_folder'))
		self.output_label.pack(side=tk.LEFT)
		ttk.Entry(output_frame, textvariable=self.main.output_path, width=40).pack(
			side=tk.LEFT, padx=5, fill=tk.X, expand=True)
		self.browse_output_button = ttk.Button(
			output_frame, text=self.main.tr('browse'), command=self.main.browse_output)
		self.browse_output_button.pack(side=tk.RIGHT)

		# Stacking options
		options_frame = ttk.LabelFrame(left_frame, text=self.main.tr('options'))
		options_frame.pack(fill=tk.X, pady=10)

		# Stacking method
		stacking_method_frame = ttk.Frame(options_frame)
		stacking_method_frame.pack(fill=tk.X, pady=5)

		self.stacking_method_label = ttk.Label(
			stacking_method_frame, text=self.main.tr('stacking_method'))
		self.stacking_method_label.pack(side=tk.LEFT)
		stacking_combo = ttk.Combobox(
			stacking_method_frame, textvariable=self.main.stacking_mode, width=15)
		stacking_combo['values'] = (
			'mean', 'median', 'kappa-sigma', 'winsorized-sigma')
		stacking_combo.pack(side=tk.LEFT, padx=5)

		# Kappa value
		self.kappa_label = ttk.Label(
			stacking_method_frame, text=self.main.tr('kappa_value'))
		self.kappa_label.pack(side=tk.LEFT, padx=10)
		ttk.Spinbox(stacking_method_frame, from_=1.0, to=5.0, increment=0.1,
					textvariable=self.main.kappa, width=8).pack(side=tk.LEFT)

		# Batch size
		batch_frame = ttk.Frame(options_frame)
		batch_frame.pack(fill=tk.X, pady=5)

		self.batch_size_label = ttk.Label(
			batch_frame, text=self.main.tr('batch_size'))
		self.batch_size_label.pack(side=tk.LEFT)
		ttk.Spinbox(batch_frame, from_=0, to=500, increment=1,
					textvariable=self.main.batch_size, width=8).pack(side=tk.LEFT, padx=5)

		# Processing mode
		processing_mode_frame = ttk.LabelFrame(
			left_frame, text="Processing Mode")
		processing_mode_frame.pack(fill=tk.X, pady=10)

		 # Option for traditional stacking
		self.traditional_check = ttk.Checkbutton(processing_mode_frame, text=self.main.tr('use_traditional_mode'),
												 variable=self.main.use_traditional)
		self.traditional_check.pack(side=tk.LEFT, padx=20)
		

		# Watch mode option
		#self.watch_mode_check = ttk.Checkbutton(processing_mode_frame, 
		#                                       text="Watch mode (process new files)",
		#                                       variable=self.main.watch_mode)
		#self.watch_mode_check.pack(side=tk.LEFT, padx=20)

		# Alignment and reference options
		self.alignment_frame = ttk.LabelFrame(
			left_frame, text=self.main.tr('alignment'))
		self.alignment_frame.pack(fill=tk.X, pady=10)

		# Reference image
		ref_frame = ttk.Frame(self.alignment_frame)
		ref_frame.pack(fill=tk.X, pady=5)

		self.reference_label = ttk.Label(
			ref_frame, text=self.main.tr('reference_image'))
		self.reference_label.pack(side=tk.LEFT, padx=5)

		ttk.Entry(ref_frame, textvariable=self.main.reference_image_path,
				  width=30).pack(side=tk.LEFT, fill=tk.X, expand=True)

		self.browse_ref_button = ttk.Button(
			ref_frame, text=self.main.tr('browse'), command=self.main.browse_reference)
		self.browse_ref_button.pack(side=tk.RIGHT, padx=5)

		# Option to remove processed images
		remove_processed_frame = ttk.Frame(self.alignment_frame)
		remove_processed_frame.pack(fill=tk.X, pady=5)

		self.remove_processed_check = ttk.Checkbutton(remove_processed_frame,
													  text=self.main.tr('remove_processed'),
													  variable=self.main.remove_aligned)
		self.remove_processed_check.pack(side=tk.LEFT, padx=5)

		# Hot pixel correction
		self.hot_pixels_frame = ttk.LabelFrame(
			left_frame, text=self.main.tr('hot_pixels_correction'))
		self.hot_pixels_frame.pack(fill=tk.X, pady=10)

		hp_check_frame = ttk.Frame(self.hot_pixels_frame)
		hp_check_frame.pack(fill=tk.X, pady=5)

		self.hot_pixels_check = ttk.Checkbutton(hp_check_frame, text=self.main.tr('perform_hot_pixels_correction'),
												variable=self.main.correct_hot_pixels)
		self.hot_pixels_check.pack(side=tk.LEFT)

		# Hot pixel parameters
		hot_params_frame = ttk.Frame(self.hot_pixels_frame)
		hot_params_frame.pack(fill=tk.X, padx=5, pady=5)

		self.hot_pixel_threshold_label = ttk.Label(
			hot_params_frame, text=self.main.tr('hot_pixel_threshold'))
		self.hot_pixel_threshold_label.pack(side=tk.LEFT)
		ttk.Spinbox(hot_params_frame, from_=1.0, to=10.0, increment=0.1,
					textvariable=self.main.hot_pixel_threshold, width=8).pack(side=tk.LEFT, padx=5)

		self.neighborhood_size_label = ttk.Label(
			hot_params_frame, text=self.main.tr('neighborhood_size'))
		self.neighborhood_size_label.pack(side=tk.LEFT, padx=10)
		ttk.Spinbox(hot_params_frame, from_=3, to=15, increment=2,
					textvariable=self.main.neighborhood_size, width=8).pack(side=tk.LEFT)

		# Progress
		self.progress_frame = ttk.LabelFrame(
			left_frame, text=self.main.tr('progress'))
		self.progress_frame.pack(fill=tk.BOTH, expand=True, pady=10)

		self.progress_bar = ttk.Progressbar(self.progress_frame, maximum=100)
		self.progress_bar.pack(fill=tk.X, pady=5)

		# Time estimation frame
		time_frame = ttk.Frame(self.progress_frame)
		time_frame.pack(fill=tk.X, pady=5)

		self.remaining_time_label = ttk.Label(
			time_frame, text=self.main.tr('estimated_time'))
		self.remaining_time_label.pack(side=tk.LEFT, padx=5)
		ttk.Label(time_frame, textvariable=self.main.remaining_time_var,
				  font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

		self.elapsed_time_label = ttk.Label(
			time_frame, text=self.main.tr('elapsed_time'))
		self.elapsed_time_label.pack(side=tk.LEFT, padx=20)
		ttk.Label(time_frame, textvariable=self.main.elapsed_time_var,
				  font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

		# Remaining files display
		remaining_files_frame = ttk.Frame(self.progress_frame)
		remaining_files_frame.pack(fill=tk.X, pady=5)

		ttk.Label(remaining_files_frame, text="Files waiting:", 
				 font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
		ttk.Label(remaining_files_frame, textvariable=self.main.remaining_files_var,
				 font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
		
		# Additional folders display
		additional_folders_frame = ttk.Frame(self.progress_frame)
		additional_folders_frame.pack(fill=tk.X, pady=5)
		
		ttk.Label(additional_folders_frame, text="Additional folders:", 
				 font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
		ttk.Label(additional_folders_frame, textvariable=self.main.additional_folders_var,
				 font=("Arial", 9)).pack(side=tk.LEFT, padx=5)

		# Status text area
		self.status_text = tk.Text(self.progress_frame, height=8, wrap=tk.WORD)
		self.status_text.pack(fill=tk.BOTH, expand=True)

		# Scrollbar for status text
		status_scrollbar = ttk.Scrollbar(
			self.status_text, orient="vertical", command=self.status_text.yview)
		status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
		self.status_text.configure(yscrollcommand=status_scrollbar.set)

		# Control buttons
		control_frame = ttk.Frame(left_frame)
		control_frame.pack(fill=tk.X, pady=10)
		self.start_button = ttk.Button(control_frame, text=self.main.tr('start'), command=self.main.start_processing)
		self.start_button.pack(side=tk.LEFT, padx=5)

		self.stop_button = ttk.Button(control_frame, text=self.main.tr('stop'), 
									 command=self.main.stop_processing, 
									 state=tk.DISABLED)
		self.stop_button.pack(side=tk.LEFT, padx=5)

		# Button to add files
		self.add_files_button = ttk.Button(control_frame, 
										  text="Add folder", 
										  command=self.main.add_folder,
										  state=tk.DISABLED)
		self.add_files_button.pack(side=tk.RIGHT, padx=5)

		# -----------------------
		# RIGHT AREA (PREVIEW)
		# -----------------------

		# Preview area
		self.preview_frame = ttk.LabelFrame(
			right_frame, text="Preview")
		self.preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)

		# Canvas to display images
		self.preview_canvas = tk.Canvas(
			self.preview_frame, bg="black", width=400, height=500)
		self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

		# Label to display current stack name
		self.current_stack_label = ttk.Label(
			self.preview_frame, text="No current stack", font=("Arial", 10, "bold"))
		self.current_stack_label.pack(pady=5)

		# Image information
		self.image_info_text = tk.Text(
			self.preview_frame, height=4, wrap=tk.WORD)
		self.image_info_text.pack(fill=tk.X, expand=False, pady=5)
		self.image_info_text.insert(
			tk.END, "Image information: waiting for processing")
		self.image_info_text.config(state=tk.DISABLED)  # Read-only

		# Stretch option for preview
		stretch_frame = ttk.Frame(self.preview_frame)
		stretch_frame.pack(fill=tk.X, pady=5)

		self.stretch_check = ttk.Checkbutton(stretch_frame, text="Apply display stretching",
											 variable=self.main.apply_stretch,
											 command=self.main.refresh_preview)
		self.stretch_check.pack(side=tk.LEFT, padx=5)

		# Reset zoom button
		self.reset_zoom_button = ttk.Button(stretch_frame, text="Reset Zoom",
										   command=lambda: self.main.preview_manager.reset_zoom() 
										   if hasattr(self.main, 'preview_manager') else None)
		self.reset_zoom_button.pack(side=tk.RIGHT, padx=5)  
		
	def update_ui_language(self):
		"""Update all interface elements with current language."""
		# Update window title
		self.main.root.title(self.main.tr('title'))

		# Update labels
		self.input_label.config(text=self.main.tr('input_folder'))
		self.output_label.config(text=self.main.tr('output_folder'))
		self.browse_input_button.config(text=self.main.tr('browse'))
		self.browse_output_button.config(text=self.main.tr('browse'))
		self.browse_ref_button.config(text=self.main.tr('browse'))

		# Update options
		self.alignment_frame.config(text=self.main.tr('alignment'))
		self.reference_label.config(text=self.main.tr('reference_image'))

		# Update option labels
		self.stacking_method_label.config(text=self.main.tr('stacking_method'))
		self.kappa_label.config(text=self.main.tr('kappa_value'))
		self.batch_size_label.config(text=self.main.tr('batch_size'))

		# Update progress section
		self.progress_frame.config(text=self.main.tr('progress'))
		self.remaining_time_label.config(text=self.main.tr('estimated_time'))
		self.elapsed_time_label.config(text=self.main.tr('elapsed_time'))

		# Update buttons
		self.start_button.config(text=self.main.tr('start'))
		self.stop_button.config(text=self.main.tr('stop'))

		# Update hot pixel correction section
		self.hot_pixels_frame.config(text=self.main.tr('hot_pixels_correction'))
		self.hot_pixels_check.config(
			text=self.main.tr('perform_hot_pixels_correction'))
		self.hot_pixel_threshold_label.config(
			text=self.main.tr('hot_pixel_threshold'))
		self.neighborhood_size_label.config(text=self.main.tr('neighborhood_size'))

		# Update processing mode options
		#self.queue_check.config(text=self.main.tr('use_queue_mode'))
		self.traditional_check.config(text=self.main.tr('use_traditional_mode'))
		self.remove_processed_check.config(text=self.main.tr('remove_processed'))