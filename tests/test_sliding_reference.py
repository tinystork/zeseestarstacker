import numpy as np

class Dummy:
    def __init__(self, update_ref_every):
        self.update_ref_every = update_ref_every
        self._images_since_ref = 0
        self.current_ref_image = None
        self.images_in_cumulative_stack = 0

    def _update_sliding_reference(self, aligned_img):
        if not self.update_ref_every:
            return
        self._images_since_ref += 1
        if self.current_ref_image is None:
            self.current_ref_image = aligned_img.copy()
        if self._images_since_ref >= self.update_ref_every:
            self.current_ref_image = aligned_img.copy()
            self._images_since_ref = 0


def test_no_op():
    q = Dummy(0)
    for i in range(5):
        q.images_in_cumulative_stack = i + 1
        q._update_sliding_reference(np.full((1, 1), i, dtype=np.float32))
    assert q.current_ref_image is None
    assert q._images_since_ref == 0


def test_refresh_trigger():
    q = Dummy(40)
    for i in range(120):
        q.images_in_cumulative_stack = i + 1
        arr = np.full((1, 1), i, dtype=np.float32)
        q._update_sliding_reference(arr)
        if i + 1 in (40, 80, 120):
            assert np.array_equal(q.current_ref_image, arr)


def test_skip_on_failure():
    q = Dummy(40)
    frame = 0
    for i in range(42):
        if i == 24:
            continue
        arr = np.full((1, 1), i, dtype=np.float32)
        frame += 1
        q.images_in_cumulative_stack = frame
        q._update_sliding_reference(arr)
    assert np.array_equal(q.current_ref_image, np.full((1,1),40,dtype=np.float32))
