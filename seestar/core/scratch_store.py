"""Run-scoped scratch store for the low-RAM Winsorized path (Lot D).

A scratch store owns a run-scoped directory UNDER the output folder (never
``/tmp`` tmpfs), and provides:

* a pre-write quota + ``statvfs`` free-space check so an insufficient disk is
  refused BEFORE any partial write (no partial scratch artifact),
* unique, collision-free file names (``tempfile`` in the store directory),
* float32 ``.npy`` memmap spill of already-aligned/normalised frames, and
  read-only memmap reopen,
* explicit, idempotent cleanup of ONLY the artifacts this store created
  (Windows-safe: every handle is closed before any unlink).

The store never holds on to a memmap handle it does not own: every returned
handle is registered and closed by :meth:`close_handles` / :meth:`cleanup`.

Lifecycle (mirrors the run):
    store = ScratchStore(output_folder)      # refuses when output_folder unset
    store.ensure_dir()                       # creates <output>/winsor_scratch
    store.check_space(needed)                # quota + statvfs, raises on refusal
    mmap = store.spill_image(arr, "img_0")   # writes .npy, returns read-only mmap
    ... reduce ...
    store.cleanup()                          # close handles + unlink created files
"""

from __future__ import annotations

import os
import tempfile
from typing import List, Optional

import numpy as np

# Subdirectory of the output folder that owns ALL low-RAM winsorized scratch
# artifacts (never /tmp — tmpfs would defeat the "disk-backed" purpose).
SCRATCH_SUBDIR = "winsor_scratch"

# Conservative free-space floor kept on the scratch volume beyond the requested
# bytes (the OS and other run artifacts must keep working; ``statvfs`` is the
# authority for what is actually free).
SCRATCH_QUOTA_FREE_FLOOR_BYTES = 64 * 1024 * 1024


class ScratchSpaceRefused(OSError):
    """Raised when the scratch volume cannot hold the requested bytes."""


def _statvfs_free_bytes(path: str) -> Optional[int]:
    try:
        st = os.statvfs(path)
        return int(st.f_bavail) * int(st.f_frsize)
    except OSError:
        return None


class ScratchStore:
    """One run-scoped scratch store (per reduction lifecycle)."""

    def __init__(self, output_folder: str):
        if not output_folder:
            raise ScratchSpaceRefused(
                "scratch store requires an output folder (never /tmp)"
            )
        self.output_folder = os.path.abspath(str(output_folder))
        self.dir = os.path.join(self.output_folder, SCRATCH_SUBDIR)
        # Files this store created (absolute paths); cleanup removes ONLY these.
        self._created: List[str] = []
        # Handles this store opened (closed before any unlink, Windows-safe).
        self._handles: List[np.memmap] = []

    # -- directory / quota ---------------------------------------------------

    def ensure_dir(self) -> str:
        os.makedirs(self.dir, exist_ok=True)
        return self.dir

    def free_bytes(self) -> Optional[int]:
        return _statvfs_free_bytes(self.dir if os.path.isdir(self.dir)
                                   else self.output_folder)

    def check_space(self, needed_bytes: int) -> None:
        """Refuse BEFORE any write when the volume cannot hold the request.

        ``needed_bytes`` is the total scratch this reduction will write.  The
        actual free space (``statvfs``) must cover it plus the named free-space
        floor.  Raises :class:`ScratchSpaceRefused` on refusal (never a partial
        write).
        """
        needed = int(needed_bytes)
        free = self.free_bytes()
        if free is None:
            # Cannot stat the volume: refuse conservatively rather than write
            # blind onto a possibly-full filesystem.
            raise ScratchSpaceRefused("cannot stat scratch volume free space")
        if needed + int(SCRATCH_QUOTA_FREE_FLOOR_BYTES) > free:
            raise ScratchSpaceRefused(
                f"scratch volume has {free} bytes free; needs "
                f"{needed + SCRATCH_QUOTA_FREE_FLOOR_BYTES}"
            )

    # -- artifacts -----------------------------------------------------------

    def new_path(self, name: str) -> str:
        """A unique, non-existing path inside the store for ``name``."""
        self.ensure_dir()
        fd, path = tempfile.mkstemp(
            prefix=f".{name}_", suffix=".npy", dir=self.dir
        )
        os.close(fd)
        os.remove(path)  # open_memmap below recreates it; keep the unique name
        self._created.append(path)
        return path

    def spill_image(self, arr, name: str) -> np.memmap:
        """Write one aligned/normalised frame to scratch and reopen read-only.

        Preserves values, dtype, geometry exactly.  The returned read-only
        memmap is registered for handle-close on cleanup.  The caller is
        expected to drop its in-RAM reference to ``arr`` afterwards (the store
        does not mutate ``arr``).
        """
        path = self.new_path(name)
        mm = np.lib.format.open_memmap(
            path, mode="w+", dtype=arr.dtype, shape=arr.shape
        )
        mm[:] = arr
        mm.flush()
        self._close_handle(mm)
        ro = np.lib.format.open_memmap(path, mode="r")
        self._handles.append(ro)
        return ro

    def new_memmap(self, name: str, shape, dtype=np.float32) -> np.memmap:
        """Allocate a writable, disk-backed float array in the store.

        Used for the low-RAM SCI/WHT outputs (written per tile).  The returned
        memmap is registered for handle-close on cleanup; the file is removed
        only by :meth:`cleanup` (so the caller may keep it alive until its
        downstream consumer finishes).
        """
        path = self.new_path(name)
        mm = np.lib.format.open_memmap(
            path, mode="w+", dtype=dtype, shape=tuple(shape)
        )
        mm[:] = 0.0
        mm.flush()
        self._handles.append(mm)
        return mm

    def flush(self) -> None:
        """Flush every writable handle this store opened (durability)."""
        for handle in self._handles:
            try:
                if hasattr(handle, "flush"):
                    handle.flush()
            except Exception:
                pass

    # -- lifecycle -----------------------------------------------------------

    def _close_handle(self, handle) -> None:
        try:
            if hasattr(handle, "flush"):
                handle.flush()
            if hasattr(handle, "_mmap") and handle._mmap is not None:
                handle._mmap.close()
        except Exception:
            pass

    def close_handles(self) -> None:
        """Close every memmap handle this store opened (Windows-safe)."""
        for handle in self._handles:
            self._close_handle(handle)
        self._handles = []

    def cleanup(self) -> None:
        """Close handles, then remove ONLY the files this store created."""
        self.close_handles()
        for path in self._created:
            try:
                os.remove(path)
            except OSError:
                pass
        self._created = []
        # Remove the directory only if empty (never a shared dir).
        try:
            if os.path.isdir(self.dir) and not os.listdir(self.dir):
                os.rmdir(self.dir)
        except OSError:
            pass
