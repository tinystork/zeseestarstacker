import json
import numpy as np
from astropy.io import fits
import os
import analyse_logic
from stack_plan import generate_stacking_plan


def extract_results(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    end = max(i for i, l in enumerate(lines) if l.strip() == "--- END VISUALIZATION DATA ---")
    start = max(i for i, l in enumerate(lines[:end]) if l.strip() == "--- BEGIN VISUALIZATION DATA ---")
    json_str = "".join(lines[start + 1:end])
    return json.loads(json_str)


def test_log_updated_after_organize(tmp_path):
    input_dir = tmp_path / "in"
    output_root = tmp_path / "out"
    input_dir.mkdir()
    output_root.mkdir()

    data = np.zeros((4, 4), dtype=np.float32)
    header = fits.Header()
    header['EQMODE'] = 1
    header['TELESCOP'] = 'T1'
    header['FILTER'] = 'L'
    header['DATE-OBS'] = '2023-01-01T00:00:00'
    fits_path = input_dir / "img.fit"
    fits.writeto(fits_path, data, header)

    options = {
        'output_root': str(output_root),
        'analyze_snr': True,
        'snr_selection_mode': 'none',
        'use_bortle': False,
        'detect_trails': False,
        'analyse_fwhm': False,
        'analyse_ecc': False,
    }
    callbacks = {'status': lambda *a, **k: None,
                 'progress': lambda *a, **k: None,
                 'log': lambda *a, **k: None}

    log_file = str(tmp_path / "log.txt")
    results = analyse_logic.perform_analysis(str(input_dir), log_file, options, callbacks)
    assert len(results) == 1
    dest = results[0]['filepath_dst']

    # Organize files
    analyse_logic.apply_pending_organization(results, log_callback=lambda *a, **k: None,
                                             status_callback=lambda *a, **k: None,
                                             progress_callback=lambda *a, **k: None,
                                             input_dir_abs=str(input_dir))

    # Update log with new paths
    analyse_logic.write_log_summary(log_file, str(input_dir), options, results_list=results)

    loaded = extract_results(log_file)
    assert loaded[0]['path'] == dest
