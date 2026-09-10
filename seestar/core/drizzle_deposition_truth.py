"""ZSSS-DRIZZLE-CLOSURE-P2A: passive, bounded live deposition-truth witness.

Mission: ``zsss-drizzle-scientific-closure-p2-20260910`` (P2-A rework-1).

This module is **diagnostic only**, **opt-in** and **fail-open**.  When it is
not explicitly enabled it performs no work at all and cannot change any
science array.  When enabled it samples, for a small bounded set of target
output pixels, the *native accumulator state* immediately before and after
every real accepted frame deposition, and records the per-add signed deltas::

    D_i = WHT_after - WHT_before            (signed native-WHT delta)
    N_i = (img_after*WHT_after) - (img_before*WHT_before)   (signed numerator delta)

The per-target aggregates are

* ``sum_D``               signed native-WHT denominator sum ``Σ D_i``
* ``pos_D`` / ``neg_D``   positive / negative add deltas
* ``sum_abs_D``           ``Σ |D_i|``
* ``sum_N`` / ``pos_N`` / ``neg_N`` / ``sum_abs_N``
* ``cancellation_quality`` = ``abs(sum_D) / sum_abs_D`` (``None`` when zero)
* ``reconstructed_sci``   = ``sum_N / sum_D``

These use the *exact* production pixmap / data / weight / background /
exposure / kernel behaviour, because they are read from the real engine
accumulator between real adds; nothing is re-derived from an offline WCS.

Enablement is explicit and bounded
----------------------------------
Set ``ZSSS_DEPOSITION_TRUTH_TARGETS`` to a JSON target file before the run::

    {"targets": [{"run": "L2", "kernel": "lanczos2", "channel": 0,
                  "row": 4266, "col": 42, "category": "catastrophic_lanczos2"}]}

Optionally set ``ZSSS_DEPOSITION_TRUTH_OUT`` for the artifact path; otherwise
the caller supplies the run output directory.

Bounded contract
----------------
* at most :data:`MAX_TARGETS` targets (extra targets are dropped, reported);
* at most :data:`MAX_ROWS` audited per-add rows (aggregates are always kept);
* only scalar/index snapshots are read (never a whole accumulator or pixmap
  copy is retained);
* deterministic ordering (target-file order; deposition order);
* every public entry point is fail-open and records failures truthfully.

Hard freezes honoured: signed WHT is preserved (never ``abs``), no clipping,
no threshold, no conditioning rule, no production semantics, product version
8.4.0 unchanged.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "p2a.truth.1"

TARGETS_ENV = "ZSSS_DEPOSITION_TRUTH_TARGETS"
OUT_ENV = "ZSSS_DEPOSITION_TRUTH_OUT"
ARTIFACT_FILENAME = "drizzle_deposition_truth.json"

MAX_TARGETS = 64
MAX_ROWS = 2048

# Module-level singleton recorder (None until explicitly enabled/loaded).
_STATE = {
    "loaded": False,
    "enabled": False,
    "reason": "not_loaded",
    "targets_path": None,
    "targets": (),
    "dropped_targets": 0,
    "recorder": None,
}


def _finite(value):
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except Exception:  # noqa: BLE001
        return None


def reset() -> None:
    """Test/reload helper: forget loaded configuration and recorded rows."""
    _STATE.update(
        {
            "loaded": False,
            "enabled": False,
            "reason": "not_loaded",
            "targets_path": None,
            "targets": (),
            "dropped_targets": 0,
            "recorder": None,
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
            targets.append(
                {
                    "run": str(entry.get("run")) if entry.get("run") is not None else None,
                    "kernel": str(entry.get("kernel")) if entry.get("kernel") is not None else None,
                    "channel": int(entry["channel"]),
                    "row": int(entry["row"]),
                    "col": int(entry["col"]),
                    "category": (
                        str(entry.get("category")) if entry.get("category") is not None else None
                    ),
                }
            )
        except Exception:  # noqa: BLE001
            dropped += 1
    if not targets:
        _STATE.update({"enabled": False, "reason": "no_valid_targets"})
        return
    _STATE.update(
        {
            "enabled": True,
            "reason": "enabled",
            "targets": tuple(targets),
            "dropped_targets": dropped,
            "recorder": _Recorder(targets),
        }
    )


def is_enabled() -> bool:
    try:
        _load()
        return bool(_STATE["enabled"])
    except Exception:  # noqa: BLE001 - fail-open
        return False


def targets_for_channel(channel: int):
    if not is_enabled():
        return ()
    return tuple(t for t in _STATE["targets"] if t["channel"] == int(channel))


def dropped_targets() -> int:
    if not is_enabled():
        return 0
    return int(_STATE.get("dropped_targets", 0))


class _Recorder:
    """Bounded aggregate + audited-row recorder (scalar state only)."""

    def __init__(self, targets):
        self.targets = tuple(targets)
        self.rows = []
        self.rows_truncated = False
        self.failures = []
        self.n_adds = 0
        self.first_wht = None
        self.first_num = None
        self.last_wht = None
        self.last_num = None
        # aggregate keyed by target index
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
        self.started_ts = time.time()

    # -- scalar snapshots -------------------------------------------------
    def snapshot(self, channel: int, acc):
        """Return ``[(target_index, img, wht), ...]`` for this channel."""
        out = []
        img = acc._out_img
        wht = acc._out_wht
        for idx, t in enumerate(self.targets):
            if t["channel"] != int(channel):
                continue
            r, c = t["row"], t["col"]
            try:
                out.append((idx, float(img[r, c]), float(wht[r, c])))
            except Exception:  # noqa: BLE001 - fail-open, index out of range
                out.append((idx, None, None))
        return out

    def record(self, channel, kernel, frame_label, before, after):
        """Fold one accepted add's signed deltas into the bounded state."""
        self.n_adds += 1
        row = None
        if len(self.rows) < MAX_ROWS:
            row = {"frame": frame_label, "channel": int(channel), "kernel": kernel,
                   "deltas": []}
        for (idx, bi, bw), (_, ai, aw) in zip(before, after):
            if None in (bi, bw, ai, aw):
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
            if row is not None:
                row["deltas"].append([idx, d_i, n_i])
            # first/last cumulative state for closure
            if idx == 0:
                if self.first_wht is None:
                    self.first_wht, self.first_num = float(bw), float(bi) * float(bw)
                self.last_wht, self.last_num = float(aw), float(ai) * float(aw)
        if row is not None:
            if len(self.rows) < MAX_ROWS:
                self.rows.append(row)
            else:
                self.rows_truncated = True
        elif len(self.rows) >= MAX_ROWS:
            self.rows_truncated = True

    # -- artifact ---------------------------------------------------------
    def target_records(self, final_state=None):
        recs = []
        for idx, t in enumerate(self.targets):
            d = float(self.agg["sum_D"][idx])
            absd = float(self.agg["sum_abs_D"][idx])
            cancell = (abs(d) / absd) if absd > 0.0 else None
            sci = (float(self.agg["sum_N"][idx]) / d) if d != 0.0 else None
            rec = {
                **t,
                "n_adds": int(self.agg["n_adds"][idx]),
                "sum_D": d,
                "pos_D": float(self.agg["pos_D"][idx]),
                "neg_D": float(self.agg["neg_D"][idx]),
                "sum_abs_D": absd,
                "sum_N": float(self.agg["sum_N"][idx]),
                "pos_N": float(self.agg["pos_N"][idx]),
                "neg_N": float(self.agg["neg_N"][idx]),
                "sum_abs_N": float(self.agg["sum_abs_N"][idx]),
                "cancellation_quality": cancell,
                "reconstructed_sci": sci,
            }
            if final_state is not None and idx in final_state:
                fi, fw = final_state[idx]
                rec["final_native_img"] = fi
                rec["final_native_wht"] = fw
                rec["final_native_numerator"] = (None if (fi is None or fw is None) else fi * fw)
                if fi is not None and sci is not None:
                    rec["reconstructed_vs_native_rel_err"] = (
                        abs(sci - fi) / abs(fi) if fi != 0.0 else 0.0
                    )
            recs.append(rec)
        return recs


def snapshot(channel: int, acc):
    try:
        if not is_enabled():
            return None
        return _STATE["recorder"].snapshot(channel, acc)
    except Exception:  # noqa: BLE001 - fail-open
        return None


def record(channel, kernel, frame_label, before, after) -> None:
    try:
        if not is_enabled() or before is None or after is None:
            return
        _STATE["recorder"].record(channel, kernel, frame_label, before, after)
    except Exception as exc:  # noqa: BLE001 - fail-open
        try:
            _STATE["recorder"].failures.append(f"{type(exc).__name__}: {exc}")
        except Exception:  # noqa: BLE001
            pass


def build_artifact(accs=None, delivered_sci=None, extra=None) -> dict:
    _load()
    rec = _STATE.get("recorder")
    if not _STATE["enabled"] or rec is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "enabled": False,
            "reason": _STATE.get("reason"),
        }
    final_state = {}
    if accs is not None:
        for idx, t in enumerate(rec.targets):
            ch = t["channel"]
            try:
                acc = accs[ch]
                final_state[idx] = (
                    float(acc._out_img[t["row"], t["col"]]),
                    float(acc._out_wht[t["row"], t["col"]]),
                )
            except Exception:  # noqa: BLE001
                final_state[idx] = (None, None)
    art = {
        "schema_version": SCHEMA_VERSION,
        "enabled": True,
        "mission_id": "zsss-drizzle-scientific-closure-p2-20260910",
        "phase": "P2-A rework-1",
        "diagnostic_only": True,
        "targets_path": _STATE.get("targets_path"),
        "max_targets": MAX_TARGETS,
        "max_rows": MAX_ROWS,
        "n_targets": len(rec.targets),
        "dropped_targets": int(_STATE.get("dropped_targets", 0)),
        "n_adds_total": int(rec.n_adds),
        "rows_recorded": len(rec.rows),
        "rows_truncated": bool(rec.rows_truncated),
        "failures": list(rec.failures),
        "closure": {
            "first_wht": rec.first_wht,
            "first_numerator": rec.first_num,
            "last_wht": rec.last_wht,
            "last_numerator": rec.last_num,
        },
        "targets": rec.target_records(final_state=final_state),
        "rows": rec.rows,
    }
    if delivered_sci is not None:
        art["delivered_sci"] = delivered_sci
    if extra:
        art.update(extra)
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
    """Write the artifact next to the run output (fail-open)."""
    try:
        if not is_enabled():
            return None
        path = os.environ.get(OUT_ENV)
        if not path:
            base = out_dir or os.getcwd()
            path = os.path.join(base, ARTIFACT_FILENAME)
        art = build_artifact(accs=accs)
        return path if _atomic_write(path, art) else None
    except Exception:  # noqa: BLE001 - fail-open
        return None
