from dataclasses import dataclass

@dataclass
class Settings:
    apply_batch_feathering: bool = True

    def apply_to_ui(self, gui):
        if hasattr(gui, "apply_batch_feathering_var"):
            gui.apply_batch_feathering_var.set(self.apply_batch_feathering)
            if hasattr(gui, "_on_apply_batch_feathering_changed"):
                gui._on_apply_batch_feathering_changed()

    def collect_from_ui(self, gui):
        if hasattr(gui, "apply_batch_feathering_var"):
            self.apply_batch_feathering = bool(gui.apply_batch_feathering_var.get())

    @staticmethod
    def reset_expert_settings(gui):
        default = Settings()
        if hasattr(gui, "apply_batch_feathering_var"):
            gui.apply_batch_feathering_var.set(default.apply_batch_feathering)
            if hasattr(gui, "_on_apply_batch_feathering_changed"):
                gui._on_apply_batch_feathering_changed()
