import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import time
from seestar_stacker_process import SeestarStacker
from seestar_alignment_process import align_seestar_images_batch
from seestar_localization import Localization


class SeestarStackerGUI:
    """
    GUI for the SeestarStacker class with multilingual support.
    """
    def __init__(self):
        self.stacker = SeestarStacker()
        self.root = tk.Tk()
        
        # Initialize localization (default to English)
        self.localization = Localization('en')
        
        self.root.title(self.tr('title'))
        self.root.geometry("700x700")
        self.stacker = SeestarStacker()
        self.stacker.set_progress_callback(self.update_progress)
        self.processing = False
        self.thread = None
        self.create_layout()

    def tr(self, key):
        """Shorthand for translation lookup"""
        return self.localization.get(key)

    def create_layout(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Language selection
        language_frame = ttk.Frame(main_frame)
        language_frame.pack(fill=tk.X, pady=5)
        ttk.Label(language_frame, text="Language / Langue:").pack(side=tk.LEFT)
        self.language_var = tk.StringVar(value='en')
        language_combo = ttk.Combobox(language_frame, textvariable=self.language_var, width=15)
        language_combo['values'] = ('English', 'Français')
        language_combo.pack(side=tk.LEFT, padx=5)
        language_combo.bind('<<ComboboxSelected>>', self.change_language)

        # Input folder
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=5)
        self.input_label = ttk.Label(input_frame, text=self.tr('input_folder'))
        self.input_label.pack(side=tk.LEFT)
        self.input_path = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input_path, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.browse_input_button = ttk.Button(input_frame, text=self.tr('browse'), command=self.browse_input)
        self.browse_input_button.pack(side=tk.RIGHT)

        # Output folder
        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill=tk.X, pady=5)
        self.output_label = ttk.Label(output_frame, text=self.tr('output_folder'))
        self.output_label.pack(side=tk.LEFT)
        self.output_path = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_path, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.browse_output_button = ttk.Button(output_frame, text=self.tr('browse'), command=self.browse_output)
        self.browse_output_button.pack(side=tk.RIGHT)

        # Options
        options_frame = ttk.LabelFrame(main_frame, text=self.tr('options'))
        options_frame.pack(fill=tk.X, pady=10)

        self.stacking_method_label = ttk.Label(options_frame, text=self.tr('stacking_method'))
        self.stacking_method_label.pack(side=tk.LEFT)
        self.stacking_mode = tk.StringVar(value="kappa-sigma")
        stacking_combo = ttk.Combobox(options_frame, textvariable=self.stacking_mode, width=15)
        stacking_combo['values'] = ('mean', 'median', 'kappa-sigma', 'winsorized-sigma')
        stacking_combo.pack(side=tk.LEFT, padx=5)

        self.kappa_label = ttk.Label(options_frame, text=self.tr('kappa_value'))
        self.kappa_label.pack(side=tk.LEFT, padx=10)
        self.kappa = tk.DoubleVar(value=2.5)
        ttk.Spinbox(options_frame, from_=1.0, to=5.0, increment=0.1, textvariable=self.kappa, width=8).pack(side=tk.LEFT)

        self.batch_size_label = ttk.Label(options_frame, text=self.tr('batch_size'))
        self.batch_size_label.pack(side=tk.LEFT, padx=10)
        self.batch_size = tk.IntVar(value=0)
        ttk.Spinbox(options_frame, from_=0, to=500, increment=1, textvariable=self.batch_size, width=8).pack(side=tk.LEFT)

        # Alignment option
        self.alignment_frame = ttk.LabelFrame(main_frame, text=self.tr('alignment'))
        self.alignment_frame.pack(fill=tk.X, pady=10)
        self.perform_alignment = tk.BooleanVar(value=True)
        self.alignment_check = ttk.Checkbutton(self.alignment_frame, text=self.tr('perform_alignment'), 
                                              variable=self.perform_alignment)
        self.alignment_check.pack(anchor=tk.W)

        self.reference_image_path = tk.StringVar()
        self.reference_label = ttk.Label(self.alignment_frame, text=self.tr('reference_image'))
        self.reference_label.pack(anchor=tk.W, padx=5)
        ttk.Entry(self.alignment_frame, textvariable=self.reference_image_path, width=50).pack(fill=tk.X, padx=5)

        # Hot Pixels Correction
        self.hot_pixels_frame = ttk.LabelFrame(main_frame, text=self.tr('hot_pixels_correction'))
        self.hot_pixels_frame.pack(fill=tk.X, pady=10)

        self.correct_hot_pixels = tk.BooleanVar(value=True)
        self.hot_pixels_check = ttk.Checkbutton(self.hot_pixels_frame, text=self.tr('perform_hot_pixels_correction'), 
                                       variable=self.correct_hot_pixels)
        self.hot_pixels_check.pack(anchor=tk.W)

        # Hot pixel parameters
        hot_params_frame = ttk.Frame(self.hot_pixels_frame)
        hot_params_frame.pack(fill=tk.X, padx=5, pady=5)

        self.hot_pixel_threshold_label = ttk.Label(hot_params_frame, text=self.tr('hot_pixel_threshold'))
        self.hot_pixel_threshold_label.pack(side=tk.LEFT)
        self.hot_pixel_threshold = tk.DoubleVar(value=3.0)
        ttk.Spinbox(hot_params_frame, from_=1.0, to=10.0, increment=0.1, textvariable=self.hot_pixel_threshold, width=8).pack(side=tk.LEFT, padx=5)

        self.neighborhood_size_label = ttk.Label(hot_params_frame, text=self.tr('neighborhood_size'))
        self.neighborhood_size_label.pack(side=tk.LEFT, padx=10)
        self.neighborhood_size = tk.IntVar(value=5)
        ttk.Spinbox(hot_params_frame, from_=3, to=15, increment=2, textvariable=self.neighborhood_size, width=8).pack(side=tk.LEFT)

        # Progress
        self.progress_frame = ttk.LabelFrame(main_frame, text=self.tr('progress'))
        self.progress_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)

        # Time estimates frame
        time_frame = ttk.Frame(self.progress_frame)
        time_frame.pack(fill=tk.X, pady=5)
        
        self.remaining_time_label = ttk.Label(time_frame, text=self.tr('estimated_time'))
        self.remaining_time_label.pack(side=tk.LEFT, padx=5)
        self.remaining_time_var = tk.StringVar(value="--:--:--")
        ttk.Label(time_frame, textvariable=self.remaining_time_var, font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        self.elapsed_time_label = ttk.Label(time_frame, text=self.tr('elapsed_time'))
        self.elapsed_time_label.pack(side=tk.LEFT, padx=20)
        self.elapsed_time_var = tk.StringVar(value="00:00:00")
        ttk.Label(time_frame, textvariable=self.elapsed_time_var, font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

        self.status_text = tk.Text(self.progress_frame, height=10, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True)

        # Control Buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)

        self.start_button = ttk.Button(control_frame, text=self.tr('start'), command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(control_frame, text=self.tr('stop'), command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Timer variables
        self.start_time = None
        self.timer_id = None

    def change_language(self, event=None):
        """Change l'interface à la langue sélectionnée"""
        selected = self.language_var.get()
        
        if selected == 'English':
            self.localization.set_language('en')
        elif selected == 'Français':
            self.localization.set_language('fr')
        
        # Update all UI elements with new language
        self.update_ui_language()

    def update_ui_language(self):
        """Met à jour tous les éléments de l'interface avec la langue actuelle"""
        # Update window title
        self.root.title(self.tr('title'))
        
        # Update labels
        self.input_label.config(text=self.tr('input_folder'))
        self.output_label.config(text=self.tr('output_folder'))
        self.browse_input_button.config(text=self.tr('browse'))
        self.browse_output_button.config(text=self.tr('browse'))
        
        # Update options
        self.alignment_frame.config(text=self.tr('alignment'))
        self.alignment_check.config(text=self.tr('perform_alignment'))
        self.reference_label.config(text=self.tr('reference_image'))
        
        # Update option labels
        self.stacking_method_label.config(text=self.tr('stacking_method'))
        self.kappa_label.config(text=self.tr('kappa_value'))
        self.batch_size_label.config(text=self.tr('batch_size'))
        
        # Update progress section
        self.progress_frame.config(text=self.tr('progress'))
        self.remaining_time_label.config(text=self.tr('estimated_time'))
        self.elapsed_time_label.config(text=self.tr('elapsed_time'))
        
        # Update buttons
        self.start_button.config(text=self.tr('start'))
        self.stop_button.config(text=self.tr('stop'))

        # Update hot pixels correction section
        self.hot_pixels_frame.config(text=self.tr('hot_pixels_correction'))
        self.hot_pixels_check.config(text=self.tr('perform_hot_pixels_correction'))
        self.hot_pixel_threshold_label.config(text=self.tr('hot_pixel_threshold'))
        self.neighborhood_size_label.config(text=self.tr('neighborhood_size'))

    def browse_input(self):
        folder = filedialog.askdirectory()
        if folder:
            self.input_path.set(folder)

    def browse_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_path.set(folder)

    def update_progress(self, message, progress=None):
        if progress is not None:
            self.progress_var.set(progress)
            
            # Extract remaining time if present in the message
            if "Temps restant estimé:" in message:
                try:
                    remaining_time = message.split("Temps restant estimé:")[1].strip()
                    self.remaining_time_var.set(remaining_time)
                except:
                    pass
                    
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()

    def update_timer(self):
        if self.start_time and self.processing:
            elapsed = time.time() - self.start_time
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.elapsed_time_var.set(f"{hours:02}:{minutes:02}:{seconds:02}")
            self.timer_id = self.root.after(1000, self.update_timer)

    def start_processing(self):
        input_folder = self.input_path.get()
        output_folder = self.output_path.get()

        if not input_folder or not output_folder:
            messagebox.showerror(self.tr('error'), self.tr('select_folders'))
            return

        self.stacker.stacking_mode = self.stacking_mode.get()
        self.stacker.kappa = self.kappa.get()
        self.stacker.batch_size = self.batch_size.get()

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.processing = True
        
        # Start timing
        self.start_time = time.time()
        self.update_timer()
        self.remaining_time_var.set(self.tr('calculating'))

        self.thread = threading.Thread(target=self.run_processing, args=(input_folder, output_folder))
        self.thread.daemon = True
        self.thread.start()

    def stop_processing(self):
        if self.processing:
            self.stacker.stop_processing = True
            self.update_progress(self.tr('stop_requested'))

    def run_processing(self, input_folder, output_folder):
        try:
            if self.perform_alignment.get():
                self.update_progress(self.tr('alignment_start'))
                # Ensure batch_size is never zero
                if align_batch_size == 0:
                    sample_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.fit', '.fits'))]
                    if sample_files:
                        sample_path = os.path.join(input_folder, sample_files[0])
                        align_batch_size = estimate_batch_size(sample_path)
                        print(f"🧠 Taille de lot dynamique estimée via GUI : {align_batch_size}")
                    else:
                        align_batch_size = 10
                # Ensure neighborhood_size is odd
                neighborhood_size = self.neighborhood_size.get()
                if neighborhood_size % 2 == 0:
                    neighborhood_size += 1
                    self.neighborhood_size.set(neighborhood_size)
                    self.update_progress(f"⚠️ La taille du voisinage ajustée à {neighborhood_size} (doit être impaire)")

                aligned_folder = align_seestar_images_batch(
                    input_folder=input_folder,
                    bayer_pattern="GRBG",
                    batch_size=align_batch_size,
                    manual_reference_path=self.reference_image_path.get() or None,
                    correct_hot_pixels=self.correct_hot_pixels.get(),
                    hot_pixel_threshold=self.hot_pixel_threshold.get(),
                    neighborhood_size=neighborhood_size
                )
                
                # Utilisez le dossier aligné comme nouveau dossier d'entrée
                self.update_progress(self.tr('using_aligned_folder').format(aligned_folder))
                input_folder = aligned_folder

            # Perform stacking
            self.update_progress(self.tr('stacking_start'))
            self.stacker.stack_images(input_folder, output_folder, batch_size=self.stacker.batch_size)
        except Exception as e:
            self.update_progress(f"❌ {self.tr('error')} : {e}")
        finally:
            self.processing = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            
            # Stop the timer
            if self.timer_id:
                self.root.after_cancel(self.timer_id)
                self.timer_id = None

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = SeestarStackerGUI()
    app.run()
