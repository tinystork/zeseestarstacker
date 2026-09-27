"""C7 — resume freeze comparison + provenance (pure, no provider objects).

Resume hard-refusal (mission §26): the frozen calibration signature is compared
field-by-field; any divergence refuses the resume *naming the field*, never a
silent recalibration and never a fallback to the uncalibrated path.  Legacy
(v2, no calibration fields) runs are normalised to "calibration disabled" (never
an invented value).

Provenance (mission §28): a bounded, readable summary built from the frozen
freeze + the provider's master admissions (role + content identity).  It never
reconstructs per-frame provenance by hand — that lives in
``CalibrationResult.provenance`` and is surfaced in DEBUG, not here.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Tuple

# Order matters: a divergence is reported on the FIRST differing field.
CALIBRATION_FREEZE_FIELDS: Tuple[str, ...] = (
    "calibration_enabled",
    "calibration_provider",
    "calibration_api_version",
    "calibration_product_version",
    "calibration_library_fingerprint",
    "calibration_contract_versions",
    "calibration_plan_map",
)


def _freeze_value(freeze: Mapping[str, Any], field: str) -> Any:
    value = freeze.get(field)
    if field == "calibration_enabled":
        # Absent == "calibration disabled / legacy run" (never invented).
        return bool(value)
    return value  # absent -> None


def compare_calibration_freeze(
    checkpoint: Mapping[str, Any],
    current: Mapping[str, Any],
) -> Tuple[bool, Optional[str]]:
    """Compare the frozen calibration signature.

    Returns ``(ok, field)``.  ``ok`` is True when the two freezes are identical
    (including both-empty == both-disabled).  Otherwise ``ok`` is False and
    ``field`` names the FIRST diverging field (per §4.1 mapping).
    """
    checkpoint = dict(checkpoint or {})
    current = dict(current or {})
    for field in CALIBRATION_FREEZE_FIELDS:
        if _freeze_value(checkpoint, field) != _freeze_value(current, field):
            return False, field
    return True, None


def calibration_refusal_reason(field: str) -> str:
    """Hard-refusal reason naming the diverging field (never a silent fallback)."""
    return (
        "Calibration freeze diverged: field "
        f"'{field}' changed since the run was frozen; refusing resume "
        "(no silent recalibration, no fallback to the uncalibrated path)."
    )


def render_calibration_provenance(
    freeze: Mapping[str, Any],
    admissions: Iterable[Mapping[str, Any]] = (),
) -> str:
    """Render the bounded §28 provenance summary (readable, no giant dump).

    ``freeze`` is the frozen ``calibration`` section (7 fields); ``admissions``
    is the provider's ``SessionResult.admissions`` (role + content identity).
    """
    freeze = dict(freeze or {})
    enabled = bool(freeze.get("calibration_enabled"))
    lines = [f"calibration enabled: {enabled}"]
    if not enabled:
        lines.append("calibration: disabled (legacy / uncalibrated run)")
        return "\n".join(lines)

    provider = freeze.get("calibration_provider") or "unknown"
    api = freeze.get("calibration_api_version") or "?"
    product = freeze.get("calibration_product_version") or "?"
    fingerprint = freeze.get("calibration_library_fingerprint") or ""
    lines.append(f"provider: {provider} {product} (API {api})")
    lines.append(f"library fingerprint: {fingerprint or '<none>'}")

    by_role: dict = {}
    for adm in admissions or ():
        if not isinstance(adm, Mapping):
            continue
        role = str(adm.get("role", "?"))
        sha = str(adm.get("content_sha256", ""))[:8]
        by_role.setdefault(role, []).append(sha)
    if by_role:
        masters = " ".join(
            f"{role}:{','.join(shas)}" for role, shas in sorted(by_role.items())
        )
        lines.append(f"masters bound: {masters}")
    else:
        lines.append("masters bound: <none>")

    plan_map = freeze.get("calibration_plan_map") or {}
    flat_applied = False
    additive_states: set = set()
    roles: set = set()
    levels: set = set()
    for entry in plan_map.values():
        if not isinstance(entry, dict):
            continue
        comp = entry.get("composition") or {}
        if comp.get("flat_applied"):
            flat_applied = True
        additive = comp.get("additive_state")
        if additive:
            additive_states.add(str(additive))
        for r in comp.get("applied_roles") or ():
            roles.add(str(r))
        lvl = comp.get("level")
        if lvl:
            levels.add(str(lvl))

    lines.append(f"flat applied: {flat_applied}")
    lines.append(
        "additive applied: "
        + (", ".join(sorted(additive_states)) if additive_states else "none")
    )
    if roles:
        lines.append(f"applied roles: {', '.join(sorted(roles))}")
    if levels:
        lines.append(f"level: {', '.join(sorted(levels))}")
    lines.append(f"plans: {len(plan_map)} light signature(s)")
    return "\n".join(lines)


__all__ = [
    "CALIBRATION_FREEZE_FIELDS",
    "calibration_refusal_reason",
    "compare_calibration_freeze",
    "render_calibration_provenance",
]
