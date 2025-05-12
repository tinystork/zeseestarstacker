# --- START OF FILE seestar/gui/ui_utils.py ---
import tkinter as tk

class ToolTip:
    """Crée une infobulle pour un widget donné."""
    def __init__(self, widget, text_callback):
        self.widget = widget
        self.text_callback = text_callback
        self.tooltip_window = None
        self.id = None
        self.x = self.y = 0
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        if self.widget.winfo_exists():
            try: # Protection supplémentaire pour after
                self.id = self.widget.after(500, self.showtip)
            except tk.TclError: # Au cas où le widget est détruit juste avant after
                self.id = None


    def unschedule(self):
        id_ = self.id
        self.id = None
        if id_:
            try:
                if self.widget.winfo_exists():
                    self.widget.after_cancel(id_)
            except tk.TclError: pass

    def showtip(self):
        if self.tooltip_window or not self.widget.winfo_exists():
            return
        try:
            x_root, y_root = self.widget.winfo_rootx(), self.widget.winfo_rooty()
            y_offset = self.widget.winfo_height() + 5
        except tk.TclError:
            self.hidetip(); return

        x = x_root + 10 
        y = y_root + y_offset

        if not self.widget.winfo_exists():
            self.hidetip(); return

        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{int(x)}+{int(y)}")

        try:
            tooltip_text = self.text_callback()
            label = tk.Label(tw, text=tooltip_text, justify=tk.LEFT,
                             background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                             wraplength=400) # Augmenté wraplength
            label.pack(ipadx=1)
        except Exception as e:
            print(f"Erreur obtention/affichage texte infobulle: {e}")
            self.hidetip()

    def hidetip(self):
        tw = self.tooltip_window
        self.tooltip_window = None
        if tw:
            try:
                if tw.winfo_exists():
                    # Utiliser after(0, ...) pour éviter problèmes si appelé pendant event Tkinter
                    tw.after(0, tw.destroy) 
            except tk.TclError: pass
# --- END OF FILE seestar/gui/ui_utils.py ---