import re

def test_boring_thread_progress_regex_matches_timestamp():
    line = "2024-01-01 00:00:00 [INFO] worker: [42%] processing"
    m = re.search(r"(?:\[(\d+(?:\.\d+)?)%\]|(\d+(?:\.\d+)?)%)", line)
    assert m
    pct = float(next(filter(None, m.groups())))
    assert pct == 42.0
