import ast
import importlib.util
import sys
import types
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
dup_path = ROOT / "duplicate"
duplicate_src = dup_path.read_text()
duplicate = types.ModuleType("duplicate")
exec(compile(duplicate_src, str(dup_path), "exec"), duplicate.__dict__)
DuplicateDefFinder = duplicate.DuplicateDefFinder


def run_finder(src: str):
    tree = ast.parse(textwrap.dedent(src))
    finder = DuplicateDefFinder()
    finder.visit(tree)
    return finder.duplicates()


def test_top_level_function_duplicate():
    src = """
    def foo():
        pass

    def foo():
        pass
    """
    dups = run_finder(src)
    assert {"foo"} == set(dups)


def test_functions_in_different_scopes_not_duplicate():
    src = """
    def foo():
        pass

    def bar():
        def foo():
            pass
    """
    assert run_finder(src) == {}


def test_nested_function_duplicate():
    src = """
    def outer():
        def inner():
            pass
        def inner():
            pass
    """
    dups = run_finder(src)
    assert {"outer.inner"} == set(dups)


def test_methods_ignored():
    src = """
    class A:
        def foo(self):
            pass
        def foo(self):
            pass
    """
    assert run_finder(src) == {}


def test_class_duplicate():
    src = """
    class A:
        pass

    class A:
        pass
    """
    dups = run_finder(src)
    assert {"A"} == set(dups)
