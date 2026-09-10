"""ZSSS-DRIZZLE-CLOSURE-P2A: passive, bounded live deposition-truth witness.

Mission: ``zsss-drizzle-scientific-closure-p2-20260910`` (P2-A rework-2).

This module is **diagnostic only**, **opt-in** and **fail-open**.  When it is
not explicitly enabled it performs no work at all and cannot change any
science array.  When enabled it samples, for a small bounded set of target
output pixels, the *native accumulator state* immediately before and after
every real accepted frame deposition, and records the per-add signed deltas::

    D_i = WHT_after - WHT_before            (signed native-WHT delta)
    N_i = (img_after*WHT_after) - (img_before*WHT_before)   (signed numerator delta)

Per-target aggregates:

* ``sum_D`` / ``pos_D`` / ``neg_D`` / ``sum_abs_D``
* ``sum_N`` / ``pos_N`` / ``neg_N`` / ``sum_abs_N``
* ``cancellation_quality`` = ``abs(sum_D) / sum_abs_D`` (``None`` when zero)
* ``reconstructed_sci``   = ``sum_N / sum_D``

Run-scoped lifecycle (rework-2)
-------------------------------
A **fresh recorder** is bound for every stack run by
:func:`start_run`, which is invoked from the canonical new-run diagnostics
seam.  Repeated persistence/finalization never resets the recorder, and
sequential runs inside one process never contaminate each other.

Enablement is explicit and bounded
----------------------------------
Set ``ZSSS_DEPOSITION_TRUTH_TARGETS`` to a JSON target file before the run::

    {"targets": [{"run": "L2", "kernel": "lanczos2", "channel": 0,
                  "row": 4266, "col": 42, "category": "catastrophic_lanczos2"}]}

Targets are selected **once per run** against the run's actual effective
kernel: an entry is selected when ``target.kernel == effective_kernel`` or when
``target.kernel`` is absent/``null`` (a documented wildcard).  Every other
entry is counted in ``skipped_by_kernel`` — sampled data is never silently
relabelled with a kernel it was not sampled under.

``ZSSS_DEPOSITION_TRUTH_OUT`` optionally overrides the artifact path;
otherwise the artifact is written next to the run output
(``<out_dir>/drizzle_deposition_truth.json``), which is what makes three
sequential kernel runs in one process safe (distinct output folders).

Bounded contract
----------------
* at most :data:`MAX_TARGETS` targets (extra entries dropped and counted);
* at most :data:`MAX_ROWS` audited per-add rows (compact ``numpy`` arrays of
  ``(target_index, D_i, N_i)``); aggregates are always retained; truncation is
  recorded and never silent;
* only scalar/index snapshots are read (no accumulator/pixmap copy retained);
* deterministic ordering (target-file order, deposition order);
* :func:`measure_state_bytes` reports an **honest, recursively measured**
  in-process state size (no "≈ floats" estimate);
* every public entry point is fail-open and records failures truthfully.

Hard freezes honoured: signed WHT is preserved (never ``abs``), no clipping,
no threshold, no conditioning rule, no geometry/pixfrac work, no kernel
substitution, no production semantics, product version 8.4.0 unchanged.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import tempfile
import time

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "p2a.truth.2"

TARGETS_ENV = "ZSSS_DEPOSITION_TRUTH_TARGETS"
OUT_ENV = "ZSSS_DEPOSITION_TRUTH_OUT"
ARTIFACT_FILENAME = "drizzle_deposition_truth.json"

MAX_TARGETS = 64
MAX_ROWS = 2048

_STATE = {
    "loaded": False,
    "enabled": False,
    "reason": "not_loaded",
    "targets_path": None,
    "targets": (),
    "dropped_targets": 0,
    "recorder": None,
    "run_meta": None,
}


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def reset() -> None:
    """Test/reload helper: forget loaded configuration and any bound run."""
    _STATE.update(
        {
            "loaded": False,
            "enabled": False,
            "reason": "not_loaded",
            "targets_path": None,
            "targets": (),
            "dropped_targets": 0,
            "recorder": None,
            "run_meta": None,
        }
    )


def _load() -> None:
    if _STATE["loaded"]:
        return
    _STATE["loaded"] = True
    path = os.environ.get(TARGETS_ENV)
    if not path:
        _STATE.update({"enabled": False, "reason": "targets_env_absent"})
        return
    _STATE["targets_path"] = path
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:  # noqa: BLE001 - fail-open
        _STATE.update({"enabled": False, "reason": f"targets_load_failed:{type(exc).__name__}"})
        logger.debug("deposition-truth targets load failed (non-fatal): %s", exc)
        return
    raw = payload.get("targets") if isinstance(payload, dict) else None
    if not isinstance(raw, list) or not raw:
        _STATE.update({"enabled": False, "reason": "no_targets"})
        return
    targets = []
    dropped = 0
    for entry in raw:
        if len(targets) >= MAX_TARGETS:
            dropped += 1
            continue
        try:
            kernel = entry.get("kernel")
            targets.append(
                {
                    "run": (str(entry["run"]) if entry.get("run") is not None else None),
                    "kernel": (str(kernel) if kernel is not None else None),
                    "channel": int(entry["channel"]),
                    "row": int(entry["row"]),
                    "col": int(entry["col"]),
                    "category": (
                        str(entry["category"]) if entry.get("category") is not None else None
                    ),
                }
            )
        except Exception:  # noqa: BLE001 - malformed entry, counted truthfully
            dropped += 1
    if not targets:
        _STATE.update({"enabled": False, "reason": "no_valid_targets"})
        return
    _STATE.update(
        {"enabled": True, "reason": "enabled", "targets": tuple(targets),
         "dropped_targets": dropped, "recorder": None, "run_meta": None}
    )


def is_enabled() -> bool:
    try:
        _load()
        return bool(_STATE["enabled"])
    except Exception:  # noqa: BLE001 - fail-open
        return False


def dropped_targets() -> int:
    if not is_enabled():
        return 0
    return int(_STATE.get("dropped_targets", 0))


# ---------------------------------------------------------------------------
# run lifecycle
# ---------------------------------------------------------------------------


def start_run(kernel=None, scale=None, resume=None, out_dir=None, run_token=None):
    """Bind a fresh run-scoped recorder (call once per stack run).

    Never resets a recorder that is still being persisted; the recorder is
    replaced only when a new run starts.
    """
    try:
        _load()
        if not _STATE["enabled"]:
            return None
        eff_kernel = None if kernel is None else str(kernel)
        selected = []
        skipped_by_kernel = 0
        for t in _STATE["targets"]:
            if t["kernel"] is None:
                selected.append(t)
            elif eff_kernel is not None and t["kernel"] == eff_kernel:
                selected.append(t)
            else:
                skipped_by_kernel += 1
        meta = {
            "run_token": (None if run_token is None else str(run_token)),
            "out_dir": (None if out_dir is None else str(out_dir)),
            "effective_kernel": eff_kernel,
            "scale": (None if scale is None else float(scale)),
            "resume": (None if resume is None else bool(resume)),
            "started_ts": time.time(),
            "targets_path": _STATE.get("targets_path"),
            "config_targets": len(_STATE["targets"]),
            "selected_targets": len(selected),
            "skipped_by_kernel": int(skipped_by_kernel),
            "dropped_targets": int(_STATE.get("dropped_targets", 0)),
        }
        _STATE["recorder"] = _Recorder(selected, meta)
        _STATE["run_meta"] = meta
        logger.info(
            "M3: deposition-truth witness bound (kernel=%s selected=%d skipped_by_kernel=%d)",
            eff_kernel, len(selected), skipped_by_kernel,
        )
        return meta
    except Exception as exc:  # noqa: BLE001 - fail-open
        logger.debug("deposition-truth start_run failed (non-fatal): %s", exc)
        return None


def _recorder():
    return _STATE.get("recorder")


# ---------------------------------------------------------------------------
# recorder
# ---------------------------------------------------------------------------


def _sizeof(obj, depth=0, seen=None) -> int:
    """Conservative recursive size of a bounded state object (bytes)."""
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen or depth > 6:
        return 0
    seen.add(oid)
    size = sys.getsizeof(obj, 0)
    if isinstance(obj, np.ndarray):
        return obj.nbytes + size
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += _sizeof(k, depth + 1, seen) + _sizeof(v, depth + 1, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            size += _sizeof(v, depth + 1, seen)
    return size


class _Recorder:
    """Bounded, run-scoped aggregate + audited-row recorder (scalars only)."""

    def __init__(self, targets, meta):
        self.targets = tuple(targets)
        self.meta = dict(meta)
        n = len(self.targets)
        self.agg = {
            "sum_D": [0.0] * n,
            "pos_D": [0.0] * n,
            "neg_D": [0.0] * n,
            "sum_abs_D": [0.0] * n,
            "sum_N": [0.0] * n,
            "pos_N": [0.0] * n,
            "neg_N": [0.0] * n,
            "sum_abs_N": [0.0] * n,
            "n_adds": [0] * n,
        }
        self.initial = [None] * n      # (img, wht) first observed
        self.final = [None] * n        # (img, wht) last observed
        self.rows = []                 # compact numpy arrays
        self.rows_truncated = False
        self.failures = []
        self.kernel_mismatch_events = 0
        self.missing_state_events = 0
        self.n_adds = 0

    # -- snapshots --------------------------------------------------------
    def snapshot(self, channel: int, acc):
        out = []
        img = acc._out_img
        wht = acc._out_wht
        for idx, t in enumerate(self.targets):
            if t["channel"] != int(channel):
                continue
            try:
                out.append((idx, float(img[t["row"], t["col"]]), float(wht[t["row"], t["col"]])))
            except Exception:  # noqa: BLE001 - out-of-range target, truthful
                out.append((idx, None, None))
        return out

    # -- fold one accepted add -------------------------------------------
    def record(self, channel, kernel, frame_label, before, after):
        if before is None or after is None:
            return
        eff = self.meta.get("effective_kernel")
        if eff is not None and kernel is not None and str(kernel) != eff:
            # never silently relabel sampled data with a foreign kernel
            self.kernel_mismatch_events += 1
            return
        self.n_adds += 1
        row_arr = None
        deltas = []
        for (idx, bi, bw), (_, ai, aw) in zip(before, after):
            if None in (bi, bw, ai, aw):
                self.missing_state_events += 1
                continue
            d_i = float(aw) - float(bw)
            n_i = float(ai) * float(aw) - float(bi) * float(bw)
            a = self.agg
            a["sum_D"][idx] += d_i
            a["sum_abs_D"][idx] += abs(d_i)
            if d_i > 0.0:
                a["pos_D"][idx] += d_i
            elif d_i < 0.0:
                a["neg_D"][idx] += d_i
            a["sum_N"][idx] += n_i
            a["sum_abs_N"][idx] += abs(n_i)
            if n_i > 0.0:
                a["pos_N"][idx] += n_i
            elif n_i < 0.0:
                a["neg_N"][idx] += n_i
            a["n_adds"][idx] += 1
            if self.initial[idx] is None:
                self.initial[idx] = (float(bi), float(bw))
            self.final[idx] = (float(ai), float(aw))
            deltas.append((idx, d_i, n_i))
        if deltas and len(self.rows) < MAX_ROWS:
            row_arr = np.asarray(deltas, dtype=np.float64)
        if row_arr is not None:
            self.rows.append(
                {"frame": frame_label, "channel": int(channel), "kernel": kernel, "deltas": row_arr}
            )
        elif len(self.rows) >= MAX_ROWS:
            self.rows_truncated = True

    # -- artifact ---------------------------------------------------------
    def measure_state_bytes(self) -> int:
        return _sizeof(self.agg) + _sizeof(self.rows) + _sizeof(self.meta) + _sizeof(
            self.targets
        ) + _sizeof(self.initial) + _sizeof(self.final) + _sizeof(self.failures)

    def target_records(self, accs=None):
        recs = []
        for idx, t in enumerate(self.targets):
            d = float(self.agg["sum_D"][idx])
            absd = float(self.agg["sum_abs_D"][idx])
            n = float(self.agg["sum_N"][idx])
            init = self.initial[idx]
            fin = self.final[idx]
            rec = {
                **t,
                "n_adds": int(self.agg["n_adds"][idx]),
                "sum_D": d,
                "pos_D": float(self.agg["pos_D"][idx]),
                "neg_D": float(self.agg["neg_D"][idx]),
                "sum_abs_D": absd,
                "sum_N": n,
                "pos_N": float(self.agg["pos_N"][idx]),
                "neg_N": float(self.agg["neg_N"][idx]),
                "sum_abs_N": float(self.agg["sum_abs_N"][idx]),
                "cancellation_quality": (abs(d) / absd) if absd > 0.0 else None,
                "reconstructed_sci": (n / d) if d != 0.0 else None,
                "initial_native_img": None if init is None else init[0],
                "initial_native_wht": None if init is None else init[1],
                "final_native_img": None if fin is None else fin[0],
                "final_native_wht": None if fin is None else fin[1],
            }
            if init is not None and fin is not None:
                rec["initial_native_numerator"] = init[0] * init[1]
                rec["final_native_numerator"] = fin[0] * fin[1]
                rec["closure_sum_D_err"] = d - (fin[1] - init[1])
                rec["closure_sum_N_err"] = n - (fin[0] * fin[1] - init[0] * init[1])
                rec["initial_state_zero"] = bool(init[1] == 0.0 and init[0] * init[1] == 0.0)
                rec["resume_nonzero_initial"] = not rec["initial_state_zero"]
            recs.append(rec)
        return recs


# ---------------------------------------------------------------------------
# passive call sites
# ---------------------------------------------------------------------------


def snapshot(channel: int, acc):
    try:
        rec = _recorder()
        if not is_enabled() or rec is None:
            return None
        return rec.snapshot(channel, acc)
    except Exception:  # noqa: BLE001 - fail-open
        return None


def record(channel, kernel, frame_label, before, after) -> None:
    try:
        rec = _recorder()
        if not is_enabled() or rec is None or before is None or after is None:
            return
        rec.record(channel, kernel, frame_label, before, after)
    except Exception as exc:  # noqa: BLE001 - fail-open
        try:
            _recorder().failures.append(f"{type(exc).__name__}: {exc}")
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# artifact
# ---------------------------------------------------------------------------


def build_artifact(accs=None) -> dict:
    _load()
    rec = _recorder()
    if not _STATE["enabled"] or rec is None:
        return {"schema_version": SCHEMA_VERSION, "enabled": False,
                "reason": _STATE.get("reason") or "no_run_bound"}
    art = {
        "schema_version": SCHEMA_VERSION,
        "enabled": True,
        "mission_id": "zsss-drizzle-scientific-closure-p2-20260910",
        "phase": "P2-A rework-2",
        "diagnostic_only": True,
        "provenance": dict(rec.meta),
        "max_targets": MAX_TARGETS,
        "max_rows": MAX_ROWS,
        "n_targets": len(rec.targets),
        "n_adds_total": int(rec.n_adds),
        "kernel_mismatch_events": int(rec.kernel_mismatch_events),
        "missing_state_events": int(rec.missing_state_events),
        "rows_recorded": len(rec.rows),
        "rows_truncated": bool(rec.rows_truncated),
        "measured_state_bytes": int(rec.measure_state_bytes()),
        "failures": list(rec.failures),
        "targets": rec.target_records(accs=accs),
        "rows": [
            {
                "frame": r["frame"],
                "channel": r["channel"],
                "kernel": r["kernel"],
                "deltas": r["deltas"].tolist(),
            }
            for r in rec.rows
        ],
    }
    return art


def _atomic_write(path: str, payload: dict) -> bool:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=1, sort_keys=True)
            os.replace(tmp, path)
            return True
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:  # noqa: BLE001
                    pass
    except Exception as exc:  # noqa: BLE001 - fail-open
        logger.debug("deposition-truth artifact write failed (non-fatal): %s", exc)
        return False


def persist(out_dir: str | None, accs=None) -> str | None:
    """Write the artifact (fail-open).  Never resets the run recorder."""
    try:
        if not is_enabled() or _recorder() is None:
            return None
        path = os.environ.get(OUT_ENV)
        if not path:
            base = out_dir or (_recorder().meta.get("out_dir") if _recorder() else None) or os.getcwd()
            path = os.path.join(base, ARTIFACT_FILENAME)
        art = build_artifact(accs=accs)
        return path if _atomic_write(path, art) else None
    except Exception:  # noqa: BLE001 - fail-open
        return None
