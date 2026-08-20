"""
Module pour la fenêtre de configuration des solveurs astrométriques.

Modèle à deux solveurs (M2b) : ZeSolver (optionnel, prioritaire, fallback
ASTAP) et ASTAP (local). Les anciens solveurs ANSVR et Astrometry.net (web)
ainsi que la clé API Astrometry.net ont été retirés de l'interface.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os  # Pour les opérations sur les chemins
import platform
from seestar.core.solver_config import load_config, save_config, resolve_solver_gate
from seestar.alignment.zesolver_adapter import discover_zesolver
from .ui_utils import ToolTip

class LocalSolverSettingsWindow(tk.Toplevel):
    """
    Fenêtre de dialogue pour configurer le solveur astrométrique
    (ZeSolver optionnel / ASTAP fallback).
    """

    def tr(self, key, default=None):
        """Shortcut to parent GUI translation."""
        return self.parent_gui.tr(key, default=default)

    def __init__(self, parent_gui):
        """
        Initialise la fenêtre de configuration des solveurs locaux.

        Args:
            parent_gui: L'instance de SeestarStackerGUI parente.
        """
        super().__init__(parent_gui.root)
        self.parent_gui = parent_gui
        # Load fallback config if the parent GUI lacks one
        if not hasattr(self.parent_gui, "config"):
            try:
                self.parent_gui.config = load_config()
            except Exception:
                self.parent_gui.config = {}
        self.withdraw()  # Cacher pendant la configuration

        self.title(self.tr("solver_config_title", default="Local Astrometry Solvers Configuration"))
        self.transient(parent_gui.root)

        # --- Variables Tkinter pour les chemins et options ---
        default_solver_choice = "none"
        if hasattr(self.parent_gui.settings, 'local_solver_preference'):
            default_solver_choice = getattr(self.parent_gui.settings, 'local_solver_preference', "none")

        self.local_solver_choice_var = tk.StringVar(value=default_solver_choice)

        self.astap_path_var = tk.StringVar(
            value=getattr(self.parent_gui.settings, 'astap_path', "")
        )
        self.astap_data_dir_var = tk.StringVar(
            value=getattr(self.parent_gui.settings, 'astap_data_dir', "")
        )
        self.astap_search_radius_var = tk.DoubleVar(
            value=getattr(self.parent_gui.settings, 'astap_search_radius', 30.0)
        )

        self.astap_downsample_var = tk.IntVar(
            value=self.parent_gui.config.get('astap_default_downsample', 2)
        )

        self.astap_sensitivity_var = tk.IntVar(
            value=self.parent_gui.config.get('astap_default_sensitivity', 100)
        )

        self.cluster_threshold_var = tk.DoubleVar(
            value=self.parent_gui.config.get('cluster_panel_threshold', 0.5)
        )

        self.reproject_between_batches_var = tk.BooleanVar(
            value=getattr(
                self.parent_gui.settings, 'reproject_between_batches', False
            )
        )

        self.reproject_between_batches_var.trace_add('write', lambda *args: self._update_warning())
        self.local_solver_choice_var.trace_add('write', lambda *args: self._update_warning())

        # Construction de l'interface utilisateur
        self._build_ui()

        # Configuration finale de la fenêtre
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.update_idletasks()

        # Centrage et affichage
        self.master.update_idletasks()
        parent_x = self.master.winfo_rootx()
        parent_y = self.master.winfo_rooty()
        parent_width = self.master.winfo_width()
        parent_height = self.master.winfo_height()
        self.update_idletasks()
        self_width = self.winfo_reqwidth()
        self_height = self.winfo_reqheight()

        position_x = parent_x + (parent_width // 2) - (self_width // 2)
        position_y = parent_y + (parent_height // 2) - (self_height // 2)
        self.geometry(f"+{position_x}+{position_y}")

        self.deiconify()
        self.focus_force()
        self.grab_set()

        self._on_solver_choice_change()

    # ------------------------------------------------------------------
    # ZeSolver status (display only, cheap probe — no catalog/GPU/network)
    # ------------------------------------------------------------------
    def _zesolver_status(self):
        """Return ``(state_value, human_text)`` for the ZeSolver status label."""
        try:
            discovery = discover_zesolver()
            state = getattr(discovery.state, "value", str(discovery.state))
        except Exception as exc:
            return ("unavailable", f"ZeSolver: unavailable ({type(exc).__name__})")
        if state == "available":
            text = "ZeSolver: available"
            pv = getattr(discovery, "product_version", None)
            if pv:
                text += f" (v{pv})"
        else:
            text = f"ZeSolver: {state}"
            msg = getattr(discovery, "message", None)
            if msg:
                text += f" \u2014 {msg}"
        return (state, text)

    def _zesolver_available(self):
        return self._zesolver_status()[0] == "available"

    def _on_solver_choice_change(self, *args):
        """
        Appelée lorsque le choix du solveur (Radiobutton) change.
        Active ou désactive le cadre de configuration ASTAP.
        """
        choice = self.local_solver_choice_var.get()

        # ASTAP est configurable pour "astap" (primaire) et "zesolver" (fallback).
        astap_state = tk.DISABLED
        if choice in ("astap", "zesolver"):
            astap_state = tk.NORMAL

        if hasattr(self, 'astap_frame') and self.astap_frame.winfo_exists():
            for widget in self.astap_frame.winfo_children():
                self._set_widget_state_recursive(widget, astap_state)

        self._update_warning()

    def _set_widget_state_recursive(self, widget, state):
        """
        Change récursivement l'état d'un widget et de ses enfants (si applicable).
        """
        try:
            if 'state' in widget.configure():
                widget.configure(state=state)
        except tk.TclError:
            pass

        if hasattr(widget, 'winfo_children'):
            for child in widget.winfo_children():
                self._set_widget_state_recursive(child, state)

    def _update_warning(self, *args):
        show = False
        if self.reproject_between_batches_var.get():
            choice = self.local_solver_choice_var.get()
            astap_configured = bool(self.astap_path_var.get().strip())
            allowed, _ = resolve_solver_gate(
                choice, self._zesolver_available(), astap_configured
            )
            show = not allowed
        self.warning_label.configure(
            text='⚠️ Aucun solveur astrométrique configuré' if show else ''
        )

    def _build_ui(self):
        """
        Construit les widgets de l'interface utilisateur pour cette fenêtre.
        """
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        self.solver_choice_frame = ttk.LabelFrame(
            main_frame,
            text=self.tr("solver_label", default="Solver"),
            padding="10",
        )
        self.solver_choice_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Radiobutton(
            self.solver_choice_frame,
            text=self.tr("solver_none", default="No solver"),
            variable=self.local_solver_choice_var,
            value="none",
            command=self._on_solver_choice_change,
        ).pack(anchor=tk.W, pady=2)

        ttk.Radiobutton(
            self.solver_choice_frame,
            text=self.tr("solver_astap", default="ASTAP"),
            variable=self.local_solver_choice_var,
            value="astap",
            command=self._on_solver_choice_change,
        ).pack(anchor=tk.W, pady=2)

        ttk.Radiobutton(
            self.solver_choice_frame,
            text=self.tr("solver_zesolver", default="ZeSolver (ASTAP fallback)"),
            variable=self.local_solver_choice_var,
            value="zesolver",
            command=self._on_solver_choice_change,
        ).pack(anchor=tk.W, pady=2)

        # --- ZeSolver status display (read-only) ---
        zesolver_state, zesolver_text = self._zesolver_status()
        self.zesolver_status_label = ttk.Label(
            main_frame,
            text=zesolver_text,
            foreground="green" if zesolver_state == "available" else "orange",
        )
        self.zesolver_status_label.pack(anchor=tk.W, pady=(0, 10))

        self.astap_frame = ttk.LabelFrame(
            main_frame,
            text=self.tr("solver_astap", default="ASTAP"),
            padding="10",
        )
        self.astap_frame.pack(fill=tk.X, padx=5, pady=5)

        astap_path_sub = ttk.Frame(self.astap_frame)
        astap_path_sub.pack(fill=tk.X, pady=(5, 2))
        ttk.Label(astap_path_sub, text=self.tr("astap_exe_label", default="Executable:")).pack(side=tk.LEFT)
        ttk.Entry(astap_path_sub, textvariable=self.astap_path_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0)
        )
        ttk.Button(
            astap_path_sub,
            text=self.tr("browse", default="Browse..."),
            command=self._browse_astap_path,
            width=12,
        ).pack(side=tk.RIGHT, padx=(5, 0))


        astap_data_sub = ttk.Frame(self.astap_frame)
        astap_data_sub.pack(fill=tk.X, pady=(2, 5))
        ttk.Label(astap_data_sub, text=self.tr("astap_data_label", default="Data Dir:")).pack(side=tk.LEFT)
        ttk.Entry(astap_data_sub, textvariable=self.astap_data_dir_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0)
        )
        ttk.Button(
            astap_data_sub,
            text=self.tr("browse", default="Browse..."),
            command=self._browse_astap_data_dir,
            width=12,
        ).pack(side=tk.RIGHT, padx=(5, 0))

        astap_radius_sub = ttk.Frame(self.astap_frame)
        astap_radius_sub.pack(fill=tk.X, pady=(2, 5))
        ttk.Label(
            astap_radius_sub,
            text=self.tr(
                "astap_search_radius_label",
                default="ASTAP Search Radius (deg):",
            ),
            width=35,
            anchor="w",
        ).pack(side=tk.LEFT, padx=(0, 5))
        radius_sb = ttk.Spinbox(
            astap_radius_sub,
            from_=0.1,
            to=90.0,
            increment=0.5,
            textvariable=self.astap_search_radius_var,
            width=6,
            format="%.1f",
        )
        radius_sb.pack(side=tk.LEFT)
        ToolTip(radius_sb, lambda: self.tr("tooltip_astap_search_radius"))

        astap_down_sub = ttk.Frame(self.astap_frame)
        astap_down_sub.pack(fill=tk.X, pady=(2, 5))
        ttk.Label(
            astap_down_sub,
            text=self.tr(
                "local_solver_astap_downsample_label",
                default="Downsample:",
            ),
            width=35,
            anchor="w",
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Spinbox(
            astap_down_sub,
            from_=1,
            to=8,
            increment=1,
            textvariable=self.astap_downsample_var,
            width=6,
        ).pack(side=tk.LEFT)

        astap_sens_sub = ttk.Frame(self.astap_frame)
        astap_sens_sub.pack(fill=tk.X, pady=(2, 5))
        ttk.Label(
            astap_sens_sub,
            text=self.tr(
                "local_solver_astap_sens_label",
                default="Sensitivity:",
            ),
            width=35,
            anchor="w",
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Spinbox(
            astap_sens_sub,
            from_=10,
            to=1000,
            increment=5,
            textvariable=self.astap_sensitivity_var,
            width=6,
        ).pack(side=tk.LEFT)

        self.warning_label = ttk.Label(
            main_frame,
            foreground="red",
        )

        self.warning_label.pack(anchor=tk.W)

        self._update_warning()

        button_frame = ttk.Frame(main_frame, padding="5")
        button_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))
        cancel_button = ttk.Button(
            button_frame,
            text=self.tr("cancel_button", default="Cancel"),
            command=self._on_cancel,
        )
        cancel_button.pack(side=tk.RIGHT, padx=(5, 0))
        ok_button = ttk.Button(
            button_frame,
            text=self.tr("ok_button", default="OK"),
            command=self._on_ok,
        )
        ok_button.pack(side=tk.RIGHT)

    def _browse_astap_path(self):
        initial_dir = ""
        current_path = self.astap_path_var.get()
        if current_path and os.path.exists(os.path.dirname(current_path)):
            initial_dir = os.path.dirname(current_path)
        elif os.path.exists(current_path):
             initial_dir = os.path.dirname(current_path)

        file_types = [(self.tr("executable_files", default="Executable Files"), "*.*")]
        if os.name == 'nt':
            file_types = [
                (self.tr("astap_executable_win", default="ASTAP Executable"), "*.exe"),
                (self.tr("all_files", default="All Files"), "*.*"),
            ]
        elif platform.system() == "Darwin":
            file_types = [
                (self.tr("astap_app", default="ASTAP Application"), "*.app"),
                (self.tr("all_files", default="All Files"), "*.*"),
            ]

        filepath = filedialog.askopenfilename(
            title=self.tr("select_astap_executable_title", default="Select ASTAP Executable"),
            initialdir=initial_dir if initial_dir else os.path.expanduser("~"),
            filetypes=file_types,
            parent=self
        )
        if filepath:
            self.astap_path_var.set(filepath)

    def _browse_astap_data_dir(self):
        initial_dir = self.astap_data_dir_var.get()
        if not initial_dir or not os.path.isdir(initial_dir):
            initial_dir = os.path.expanduser("~")

        dirpath = filedialog.askdirectory(
            title=self.tr("select_astap_data_dir_title", default="Select ASTAP Star Index Data Directory"),
            initialdir=initial_dir,
            parent=self
        )
        if dirpath:
            self.astap_data_dir_var.set(dirpath)

    def _on_ok(self):
        """
        Appelé lorsque l'utilisateur clique sur OK.
        Sauvegarde les paramètres et ferme la fenêtre.
        """
        solver_choice = self.local_solver_choice_var.get()
        astap_path = self.astap_path_var.get().strip()
        astap_data_dir = self.astap_data_dir_var.get().strip()
        astap_radius = self.astap_search_radius_var.get()
        astap_downsample = self.astap_downsample_var.get()
        astap_sensitivity = self.astap_sensitivity_var.get()
        self.parent_gui.settings.astap_downsample = astap_downsample
        self.parent_gui.settings.astap_sensitivity = astap_sensitivity
        cluster_threshold = self.cluster_threshold_var.get()
        reproject_batches = self.reproject_between_batches_var.get()

        # Valider que si ASTAP est choisi, son chemin principal est rempli.
        validation_ok = True
        if solver_choice == "astap" and not astap_path:
            messagebox.showerror(self.tr("error"),
                                 self.tr("astap_path_required_error", default="ASTAP is selected, but the executable path is missing."),
                                 parent=self)
            validation_ok = False

        if not validation_ok:
            return

        setattr(self.parent_gui.settings, 'local_solver_preference', solver_choice)
        self.parent_gui.settings.astap_path = astap_path
        self.parent_gui.settings.astap_data_dir = astap_data_dir
        setattr(self.parent_gui.settings, 'astap_search_radius', astap_radius)

        self.parent_gui.settings.reproject_between_batches = reproject_batches
        if hasattr(self.parent_gui, 'reproject_between_batches_var'):
            try:
                self.parent_gui.reproject_between_batches_var.set(reproject_batches)
            except Exception:
                pass
        # Refresh Add Folder button state in the main GUI if available
        try:
            if hasattr(self.parent_gui, 'update_add_folder_button_state'):
                self.parent_gui.update_add_folder_button_state()
        except Exception:
            pass

        self.parent_gui.config['astap_default_downsample'] = int(astap_downsample)
        self.parent_gui.config['astap_default_sensitivity'] = int(astap_sensitivity)
        self.parent_gui.config['cluster_panel_threshold'] = float(cluster_threshold)
        if hasattr(self.parent_gui, 'cluster_threshold_var'):
            self.parent_gui.cluster_threshold_var.set(float(cluster_threshold))
        try:
            save_config(self.parent_gui.config)
        except Exception:
            pass

        self.grab_release()
        self.destroy()

    def _on_cancel(self):
        self.grab_release()
        self.destroy()

# --- END OF FILE seestar/gui/local_solver_gui.py ---
