import numpy as np
import cv2


def compute_similarity_transform(src_img: np.ndarray, ref_img: np.ndarray) -> np.ndarray:
    """Return 2x3 affine transform aligning ``src_img`` to ``ref_img``.

    Parameters
    ----------
    src_img : np.ndarray
        Image to transform.
    ref_img : np.ndarray
        Reference image.

    Returns
    -------
    np.ndarray
        2x3 affine matrix usable with ``cv2.warpAffine``.
    """
    src_gray = src_img if src_img.ndim == 2 else src_img[..., 1]
    ref_gray = ref_img if ref_img.ndim == 2 else ref_img[..., 1]

    src_norm = cv2.normalize(src_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ref_norm = cv2.normalize(ref_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    detector = cv2.ORB_create(500)
    kp1, des1 = detector.detectAndCompute(src_norm, None)
    kp2, des2 = detector.detectAndCompute(ref_norm, None)
    if des1 is None or des2 is None or len(kp1) < 3 or len(kp2) < 3:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    if len(matches) < 3:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if M is None:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    return M.astype(np.float32)


def apply_similarity_transform(img: np.ndarray, M: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Apply affine matrix ``M`` to ``img`` using ``output_shape`` (H, W)."""
    h, w = output_shape
    if img.ndim == 3:
        out = np.zeros((h, w, img.shape[2]), dtype=img.dtype)
        for c in range(img.shape[2]):
            out[:, :, c] = cv2.warpAffine(
                img[:, :, c],
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        return out
    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def align_batch_to_reference(batch_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    """Return ``batch_img`` aligned to ``reference_img`` by similarity transform."""
    M = compute_similarity_transform(batch_img, reference_img)
    return apply_similarity_transform(batch_img, M, reference_img.shape[:2])
