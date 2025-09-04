import os
import numpy as np
from astropy.io import fits
import seestar.core.streaming_stack as ss

def test_orientation_written_channel_last_input(tmp_path):
    files = []
    for i in range(3):
        arr = np.full((4, 6, 3), i + 1, dtype=np.float32)
        p = tmp_path / f"img{i}.npy"
        np.save(p, arr)
        files.append(str(p))

    out = ss.stack_disk_streaming(files, chunk_rows=2, parallel_io=False)
    with fits.open(out, memmap=True) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header

    expected = np.mean(
        np.stack([np.full((4, 6, 3), i + 1, dtype=np.float32) for i in range(3)], axis=0),
        axis=0,
    )
    data_cl = np.moveaxis(data, 0, -1)
    assert np.allclose(data_cl, expected)
    assert hdr['NAXIS1'] == 6 and hdr['NAXIS2'] == 4 and hdr['NAXIS3'] == 3
    os.remove(out)
