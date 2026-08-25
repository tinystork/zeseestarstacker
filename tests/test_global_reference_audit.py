"""Focused audit test: the global-alignment reference only *mutates* (is
reassigned to a stacked/cumulative image) inside **positive**
``reproject_between_batches`` branches.

RF-1 corrective C.  This is a **structural path guard**, not experimental proof:
it pins the *lexical* fact that every reference mutation lives inside a
positively-guarded ``if self.reproject_between_batches`` branch (so the plain
classic / M3-Drizzle / batch-plan-flush paths never reassign the reference to a
stack).  It does **not** execute the worker and does **not** demonstrate any
*behavioural* consequence; the behavioural evidence is the batching-dependence
POC (``research/registration_field_rotation/batch_dependence_poc.py`` and
``tests/test_batch_dependence_poc.py``), and the RF-2 worker test gate remains
in ``docs/registration_field_rotation_research.md`` §8.3.

Corrective-C fixes to the previous iteration
--------------------------------------------
1. **Branch polarity.**  ``_inside_reproject_guard`` now walks ancestors and
   counts an assignment as positively guarded **only** when it is in the
   ``body`` of an ``if`` whose test *positively* asserts
   ``reproject_between_batches``.  An assignment in an ``else``/``elif``-
   negative branch of ``if self.reproject_between_batches``, or in the body of
   ``if not self.reproject_between_batches``, is **not** accepted as positively
   guarded (unless a higher positive ancestor actually applies).  The previous
   helper falsely accepted such branches by scanning ``ast.walk(parent.test)``
   for the attribute without regard to branch polarity or negation.
2. **Exact mutation-line pin.**  The test now asserts the *exact* current
   mutation line set, so a future edit that adds/removes/relocates a reference
   mutation (or renames the variable) fails loudly instead of silently passing.
3. **Snippet tests.**  ``_inside_reproject_guard`` is unit-tested on small
   parsed snippets covering positive body, else-branch, ``not``-guard, ``and``
   conjunction, and ``elif`` cases.
"""

import ast
import pathlib

QM_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "seestar"
    / "queuep"
    / "queue_manager.py"
)

VAR = "reference_image_data_for_global_alignment"
GUARD_ATTR = "reproject_between_batches"

# Exact line numbers (in ``seestar/queuep/queue_manager.py``) of every reference
# mutation (reassignment to a stack), verified against HEAD 61291aa.
MUTATION_LINES = {6252, 6259, 6266, 6749, 6754, 6759}


def _walk_with_parents(tree):
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node
    return tree


def _assigned_names(target):
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        out = []
        for elt in target.elts:
            out.extend(_assigned_names(elt))
        return out
    return []


def _call_attr(value):
    """Terminal attribute name of a Call's func (e.g. ``_get_reference_image``)."""
    if isinstance(value, ast.Call):
        f = value.func
        if isinstance(f, ast.Attribute):
            return f.attr
        if isinstance(f, ast.Name):
            return f.id
    return None


def _test_asserts_true(test):
    """True iff ``test`` *positively* asserts ``reproject_between_batches``.

    Accepted: the bare attribute, or the attribute as a non-negated term of an
    ``and`` conjunction (e.g. ``self.reproject_between_batches and X``).
    Rejected: ``not self.reproject_between_batches``, or the attribute inside an
    ``or``, a comparison, etc.
    """

    def terms(t):
        if isinstance(t, ast.BoolOp) and isinstance(t.op, ast.And):
            for v in t.values:
                yield from terms(v)
        else:
            yield t

    return any(
        isinstance(t, ast.Attribute) and t.attr == GUARD_ATTR for t in terms(test)
    )


def _inside_reproject_guard(node):
    """True iff ``node`` is in the positive ``body`` of an ``if`` whose test
    positively asserts ``reproject_between_batches``.  Branch polarity and
    negation are respected; else/elif-negative branches and ``not``-guards do
    not count, unless a higher positive ancestor applies."""
    cur = node
    while True:
        parent = getattr(cur, "_parent", None)
        if parent is None:
            return False
        if isinstance(parent, ast.If):
            in_body = any(child is cur for child in parent.body)
            if in_body and _test_asserts_true(parent.test):
                return True
            # orelse branch, or body of a non-positive if -> keep walking up
        cur = parent


def _collect_assignments():
    tree = _walk_with_parents(ast.parse(QM_PATH.read_text(encoding="utf-8")))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(VAR in _assigned_names(t) for t in targets):
                found.append(node)
    return found


def _classify(assignments):
    init = [a for a in assignments
            if isinstance(a, ast.Assign) and isinstance(a.value, ast.Constant)
            and a.value.value is None]
    initial_ref = [a for a in assignments if _call_attr(a.value) == "_get_reference_image"]
    flush_seam = [a for a in assignments if _call_attr(a.value) == "_flush_current_batch"]
    mutations = [a for a in assignments
                 if a not in init and a not in initial_ref and a not in flush_seam]
    return init, initial_ref, flush_seam, mutations


def test_reference_initialised_once_from_get_reference_image():
    init, initial_ref, _, _ = _classify(_collect_assignments())
    assert len(init) == 1, f"expected 1 `= None` initialiser, got {len(init)}"
    assert len(initial_ref) == 1, (
        f"expected exactly 1 `_get_reference_image` assignment (immutable initial "
        f"reference), got {len(initial_ref)} at lines "
        f"{[a.lineno for a in initial_ref]}"
    )


def test_exact_mutation_line_set():
    _, _, _, mutations = _classify(_collect_assignments())
    lines = {a.lineno for a in mutations}
    assert lines == MUTATION_LINES, (
        f"reference mutation line set changed: got {sorted(lines)}, "
        f"expected {sorted(MUTATION_LINES)}"
    )


def test_all_reference_mutations_positively_gated_by_reproject():
    _, _, flush_seam, mutations = _classify(_collect_assignments())
    assert len(mutations) == len(MUTATION_LINES)
    unguarded = [a for a in mutations if not _inside_reproject_guard(a)]
    assert not unguarded, (
        "reference mutation outside a *positive* `reproject_between_batches` "
        "guard found at:\n"
        + "\n".join(f"  line {a.lineno}" for a in unguarded)
    )


def test_flush_seam_is_single_noop_reassignment():
    _, _, flush_seam, _ = _classify(_collect_assignments())
    assert len(flush_seam) == 1, f"expected 1 `_flush_current_batch` seam, got {len(flush_seam)}"
    assert not _inside_reproject_guard(flush_seam[0]), (
        "flush seam must live outside the reproject guard (it is a no-op "
        "reassignment for the classic/batch-plan path)"
    )


# --------------------------------------------------------------------------
# Snippet unit tests for the branch-aware helper
# --------------------------------------------------------------------------


def _assign_in_snippet(src, target_name="ref"):
    """Parse a snippet, return the assignment node assigning ``target_name``."""
    tree = _walk_with_parents(ast.parse(src))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(target_name in _assigned_names(t) for t in node.targets):
                return node
    raise AssertionError(f"no assignment to {target_name!r} in snippet")


def test_guard_positive_body_accepted():
    node = _assign_in_snippet("if self.reproject_between_batches:\n    ref = x\n")
    assert _inside_reproject_guard(node) is True


def test_guard_else_branch_rejected():
    node = _assign_in_snippet(
        "if self.reproject_between_batches:\n    pass\nelse:\n    ref = x\n"
    )
    assert _inside_reproject_guard(node) is False


def test_guard_negated_body_rejected():
    node = _assign_in_snippet("if not self.reproject_between_batches:\n    ref = x\n")
    assert _inside_reproject_guard(node) is False


def test_guard_and_conjunction_body_accepted():
    node = _assign_in_snippet(
        "if self.reproject_between_batches and other:\n    ref = x\n"
    )
    assert _inside_reproject_guard(node) is True


def test_guard_elif_negative_branch_rejected():
    # elif compiles to the orelse of the first If: not positively guarded by it
    node = _assign_in_snippet(
        "if self.reproject_between_batches:\n    pass\nelif other:\n    ref = x\n"
    )
    assert _inside_reproject_guard(node) is False


def test_guard_nested_positive_ancestor_accepted():
    # an assignment inside an else of an *inner* if, but inside the positive
    # body of an *outer* reproject guard, is still positively guarded
    node = _assign_in_snippet(
        "if self.reproject_between_batches:\n"
        "    if other:\n        pass\n    else:\n        ref = x\n"
    )
    assert _inside_reproject_guard(node) is True


def test_guard_no_ancestor_rejected():
    node = _assign_in_snippet("ref = x\n")
    assert _inside_reproject_guard(node) is False
