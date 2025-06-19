import os
import shutil
import time
from typing import Iterable, Callable


def move_to_stacked(paths: Iterable[str], progress_cb: Callable[[str, str | None], None] | None = None, subdir: str = "stacked") -> None:
    """Move processed RAW files to a sibling 'stacked' folder."""
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        src_dir = os.path.dirname(os.path.abspath(p))
        dst_dir = os.path.join(src_dir, subdir)
        os.makedirs(dst_dir, exist_ok=True)
        base = os.path.basename(p)
        dst = os.path.join(dst_dir, base)
        if os.path.exists(dst):
            stem, ext = os.path.splitext(base)
            dst = os.path.join(dst_dir, f"{stem}_dup_{int(time.time())}{ext}")
        try:
            same_dev = os.stat(src_dir).st_dev == os.stat(dst_dir).st_dev
            if same_dev:
                os.rename(p, dst)
            else:
                shutil.copy2(p, dst)
                os.remove(p)
            if progress_cb:
                progress_cb(f"📦 Moved to stacked: {base}", "INFO_DETAIL")
        except Exception as e:
            if progress_cb:
                progress_cb(f"⚠️ Move failed for {base}: {e}", "WARN")
