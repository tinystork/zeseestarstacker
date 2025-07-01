import sys
from pathlib import Path
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import zemosaic.zemosaic_worker as worker

class DummyQueue:
    def __init__(self):
        self.items = []
    def put(self, item):
        self.items.append(item)

def test_run_process_forwards_solver_settings(monkeypatch):
    recorded = {}
    def fake_run(*args, solver_settings=None, **kwargs):
        recorded['solver'] = solver_settings
        recorded['args_len'] = len(args)
    monkeypatch.setattr(worker, 'run_hierarchical_mosaic', fake_run)
    dummy_args = [0] * 32
    q = DummyQueue()
    worker.run_hierarchical_mosaic_process(q, *dummy_args, solver_settings_dict={'solver_choice': 'ASTROMETRY'})
    assert recorded['solver']['solver_choice'] == 'ASTROMETRY'
    assert recorded['args_len'] == 33
