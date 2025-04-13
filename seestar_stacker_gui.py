import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import time
from seestar_stacker_process import SeestarStacker
from seestar_alignment_process import align_seestar_images_batch


class SeestarStackerGUI:
    """
    GUI for the SeestarStacker class.
    """
    def __init__(self):
        self.stacker = SeestarStacker()
        self.root = tk.Tk()
        self.root.title("Seestar Stacker")
        self.root.geometry("700x700")
        self.stacker = SeestarStacker()
        self.stacker.set_progress_callback(self.update_progress)
        self.processing = False
        self.thread = None
        self.create_layout()

    def create_layout(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Input folder
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=5)
        ttk.Label(input_frame, text="Dossier d'entrée:").pack(side=tk.LEFT)
        self.input_path = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input_path, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(input_frame, text="Parcourir", command=self.browse_input).pack(side=tk.RIGHT)

        # Output folder
        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill=tk.X, pady=5)
        ttk.Label(output_frame, text="Dossier de sortie:").pack(side=tk.LEFT)
        self.output_path = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_path, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="Parcourir", command=self.browse_output).pack(side=tk.RIGHT)

        # Options
        options_frame = ttk.LabelFrame(main_frame, text="Options")
        options_frame.pack(fill=tk.X, pady=10)

        ttk.Label(options_frame, text="Méthode d'empilement:").pack(side=tk.LEFT)
        self.stacking_mode = tk.StringVar(value="kappa-sigma")
        stacking_combo = ttk.Combobox(options_frame, textvariable=self.stacking_mode, width=15)
        stacking_combo['values'] = ('mean', 'median', 'kappa-sigma', 'winsorized-sigma')
        stacking_combo.pack(side=tk.LEFT, padx=5)

        ttk.Label(options_frame, text="Valeur de Kappa:").pack(side=tk.LEFT, padx=10)
        self.kappa = tk.DoubleVar(value=2.5)
        ttk.Spinbox(options_frame, from_=1.0, to=5.0, increment=0.1, textvariable=self.kappa, width=8).pack(side=tk.LEFT)

        ttk.Label(options_frame, text="Taille du lot (0 pour auto):").pack(side=tk.LEFT, padx=10)
        self.batch_size = tk.IntVar(value=0)
        ttk.Spinbox(options_frame, from_=0, to=500, increment=1, textvariable=self.batch_size, width=8).pack(side=tk.LEFT)

        # Alignment option
        alignment_frame = ttk.LabelFrame(main_frame, text="Alignement")
        alignment_frame.pack(fill=tk.X, pady=10)
        self.perform_alignment = tk.BooleanVar(value=True)
        ttk.Checkbutton(alignment_frame, text="Effectuer l'alignement avant l'empilement", variable=self.perform_alignment).pack(anchor=tk.W)

        self.reference_image_path = tk.StringVar()
        ttk.Label(alignment_frame, text="Image de référence (laissez vide pour sélection automatique) :").pack(anchor=tk.W, padx=5)
        ttk.Entry(alignment_frame, textvariable=self.reference_image_path, width=50).pack(fill=tk.X, padx=5)

        # Progress
        progress_frame = ttk.LabelFrame(main_frame, text="Progression")
        progress_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)

        # Time estimates frame
        time_frame = ttk.Frame(progress_frame)
        time_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(time_frame, text="Temps restant estimé:").pack(side=tk.LEFT, padx=5)
        self.remaining_time_var = tk.StringVar(value="--:--:--")
        ttk.Label(time_frame, textvariable=self.remaining_time_var, font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(time_frame, text="Temps écoulé:").pack(side=tk.LEFT, padx=20)
        self.elapsed_time_var = tk.StringVar(value="00:00:00")
        ttk.Label(time_frame, textvariable=self.elapsed_time_var, font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

        self.status_text = tk.Text(progress_frame, height=10, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True)

        # Control Buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)

        self.start_button = ttk.Button(control_frame, text="Démarrer", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(control_frame, text="Arrêter", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Timer variables
        self.start_time = None
        self.timer_id = None

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
            messagebox.showerror("Erreur", "Veuillez sélectionner les dossiers d'entrée et de sortie.")
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
        self.remaining_time_var.set("Calcul en cours...")

        self.thread = threading.Thread(target=self.run_processing, args=(input_folder, output_folder))
        self.thread.daemon = True
        self.thread.start()

    def stop_processing(self):
        if self.processing:
            self.stacker.stop_processing = True
            self.update_progress("⚠️ Arrêt demandé, patientez...")

    def run_processing(self, input_folder, output_folder):
        try:
            # Perform alignment if enabled
            if self.perform_alignment.get():
                self.update_progress("⚙️ Début de l'alignement des images...")
                # Ensure batch_size is never zero
                align_batch_size = self.batch_size.get()
                if align_batch_size == 0:
                    align_batch_size = 10  # Default value if 0

                aligned_folder = align_seestar_images_batch(
                    input_folder=input_folder,
                    bayer_pattern="GRBG",
                    batch_size=align_batch_size,
                    manual_reference_path=self.reference_image_path.get() or None
                )
            
                # Utilisez le dossier aligné comme nouveau dossier d'entrée
                self.update_progress(f"✅ Utilisation du dossier aligné : {aligned_folder}")
                input_folder = aligned_folder

            # Perform stacking
            self.update_progress("⚙️ Début de l'empilement des images...")
            self.stacker.stack_images(input_folder, output_folder, batch_size=self.stacker.batch_size)
        except Exception as e:
            self.update_progress(f"❌ Erreur : {e}")
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