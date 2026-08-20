"""Tests ciblés : le filtre anti-dossier ``stacked`` doit être RELATIF au
dossier scanné, jamais basé sur le chemin absolu.

Régression identifiée sur run réel (2026-08-20) : avec un dossier d'entrée
nommé ``stacked`` (ex: /home/tristan/M16/stacked), le second filtre
(``stacked_subdir_name in Path(abs_fpath).parts``) excluait TOUS les fichiers
parce que le chemin absolu contient ``stacked`` -> file vide -> 0 image
accumulée (échec propre de la garde W-1, mais run inutilisable).

Sémantique voulue (conservée) : seul un sous-dossier ``stacked`` SITUÉ DANS le
dossier scanné est exclu (relpath). Un dossier d'entrée dont le *nom* est
``stacked`` est un dossier d'entrée normal.
"""
from __future__ import annotations

import os

from seestar.queuep.queue_manager import SeestarQueuedStacker


def _make_stackable(tmp_path, folder_name: str, n: int = 3):
    folder = tmp_path / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = folder / f"Light_M16_{i:03d}.fit"
        p.write_bytes(b"FAKE-FITS")
        paths.append(str(p))
    # Sous-dossier "stacked" DANS le dossier scanné -> doit rester exclu.
    sub = folder / "stacked"
    sub.mkdir(exist_ok=True)
    excluded = sub / "already_stacked.fit"
    excluded.write_bytes(b"FAKE-FITS")
    return folder, paths, str(excluded)


def test_input_folder_named_stacked_is_not_filtered(tmp_path):
    """Un dossier d'entrée nommé 'stacked' doit être scanné normalement."""
    folder, paths, _ = _make_stackable(tmp_path, "stacked")
    s = SeestarQueuedStacker()
    s.update_progress = lambda *a, **k: None
    added = s._add_files_to_queue(str(folder))
    assert added == len(paths)
    assert s.files_in_queue == len(paths)
    queued = list(s.queue.queue)
    for p in paths:
        assert p in queued


def test_stacked_subdir_inside_scanned_folder_still_excluded(tmp_path):
    """Le sous-dossier 'stacked' DANS le dossier scanné reste exclu (relatif)."""
    folder, paths, excluded = _make_stackable(tmp_path, "input")
    s = SeestarQueuedStacker()
    s.update_progress = lambda *a, **k: None
    added = s._add_files_to_queue(str(folder))
    assert added == len(paths)
    queued = list(s.queue.queue)
    assert excluded not in queued


def test_absolute_path_with_stacked_parent_not_filtered(tmp_path):
    """Le chemin absolu contenant 'stacked' (dossier parent) ne filtre rien."""
    root = tmp_path / "M16"
    (root / "stacked").mkdir(parents=True, exist_ok=True)
    for i in range(2):
        (root / "stacked" / f"frame_{i}.fit").write_bytes(b"FAKE-FITS")
    s = SeestarQueuedStacker()
    s.update_progress = lambda *a, **k: None
    added = s._add_files_to_queue(str(root / "stacked"))
    assert added == 2
    assert s.files_in_queue == 2
