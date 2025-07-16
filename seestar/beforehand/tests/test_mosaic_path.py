import os
import numpy as np
from astropy.io import fits
import analyse_logic


def test_mosaic_path(tmp_path):
    input_dir = tmp_path / "in"
    output_root = tmp_path / "out"
    input_dir.mkdir()
    output_root.mkdir()
    data = np.zeros((10, 10), dtype=np.float32)
    header = fits.Header()
    header['EQMODE'] = 1
    header['TELESCOP'] = 'TestScope'
    header['FILTER'] = 'L'
    header['DATE-OBS'] = '2023-08-22T00:00:00'
    fits_path = input_dir / "image_mosaic_001.fit"
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
    dst = results[0]['filepath_dst']
    assert dst.startswith(str(output_root))
    expected = os.path.join(str(output_root), 'mosaic', 'EQ', 'TestScope', '2023-08-22', 'Filter_L', 'image_mosaic_001.fit')
    assert dst == expected
