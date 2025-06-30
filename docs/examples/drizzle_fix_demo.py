"""Demonstration of the drizzle double-normalisation fix."""
import numpy as np
from drizzle.resample import Drizzle

from seestar.core.drizzle_utils import drizzle_finalize

size = (32, 32)
star_pos = (16, 16)

# create synthetic input frames
frames = []
for i in range(20):
    img = np.ones(size, dtype=np.float32)
    img[star_pos] += 1000.0
    frames.append(img)

# reference drizzle stacking (single batch)
d_ref = Drizzle(out_shape=size)
for f in frames:
    d_ref.add_image(
        f, exptime=1.0, pixmap=np.dstack(np.indices(size)[::-1]).astype(np.float32)
    )
ref_img = drizzle_finalize(d_ref.out_img, d_ref.out_wht)


# two-batch stacking without early normalisation
def run_batch(frames_batch):
    d = Drizzle(out_shape=size)
    for f in frames_batch:
        d.add_image(
            f, exptime=1.0, pixmap=np.dstack(np.indices(size)[::-1]).astype(np.float32)
        )
    return d.out_img, d.out_wht


s1, w1 = run_batch(frames[:10])
s2, w2 = run_batch(frames[10:])
res_img = drizzle_finalize(s1 + s2, w1 + w2)

print("Star flux single batch:", ref_img[star_pos])
print("Star flux two batches:", res_img[star_pos])
