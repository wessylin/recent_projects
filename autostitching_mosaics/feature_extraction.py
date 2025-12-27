import numpy as np
import matplotlib.pyplot as plt
import harris_detection as HD

# ----------------- Small Gaussian blur -----------------------------
def gaussian_kernel1d(sigma: float, radius: int | None = None):
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    if radius is None:
        radius = int(np.ceil(3 * sigma))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (xs / sigma) ** 2)
    k /= k.sum()
    return k.astype(np.float32)

def convolve1d_reflect(img: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    """Reflect-pad + 1D convolution along axis (0 or 1)."""
    r = (len(k) - 1) // 2
    if axis == 0:
        pad = np.pad(img, ((r, r), (0, 0)), mode="reflect")
        out = np.empty_like(img, dtype=np.float32)
        for y in range(img.shape[0]):
            out[y] = (pad[y:y + 2 * r + 1] * k[:, None]).sum(axis=0)
        return out
    else:
        pad = np.pad(img, ((0, 0), (r, r)), mode="reflect")
        out = np.empty_like(img, dtype=np.float32)
        for x in range(img.shape[1]):
            out[:, x] = (pad[:, x:x + 2 * r + 1] * k[None, :]).sum(axis=1)
        return out

def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    k = gaussian_kernel1d(sigma)
    tmp = convolve1d_reflect(img.astype(np.float32), k, axis=0)
    out = convolve1d_reflect(tmp, k, axis=1)
    return out


# ---------------- 40×40 patch extraction around (y,x) -----------------------------
def extract_patch_40(img: np.ndarray, y: int, x: int, win: int = 40) -> np.ndarray:
    """Return a 40x40 patch centered at (y,x). Reflect-pad if needed."""
    assert win % 2 == 0, "Window must be even (40)."
    r = win // 2
    padded = np.pad(img, ((r, r), (r, r)), mode="reflect")
    y0, x0 = y + r, x + r
    return padded[y0 - r:y0 + r, x0 - r:x0 + r].astype(np.float32)


# ----------------- Downsample 40×40 -> 8×8 using area (block) averaging --------------
def downsample_40_to_8(p40: np.ndarray) -> np.ndarray:
    """Block-average 5×5 cells → 8×8 (strict area averaging, no aliasing)."""
    assert p40.shape == (40, 40)
    p = p40.reshape(8, 5, 8, 5).mean(axis=(1, 3))
    return p.astype(np.float32)


# ----------------- Descriptor normalization -----------------------------
def normalize_descriptor(d8: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std, then L2 normalize (returns 64-D vector)."""
    v = d8.reshape(-1).astype(np.float32)
    v -= v.mean()
    std = v.std()
    if std < 1e-6:
        std = 1e-6
    v /= std
    # final L2 for stability (optional but common)
    n = np.linalg.norm(v) + 1e-6
    v /= n
    return v


# ---------------------- Build 8×8 descriptors for keypoints -----------------------------
def build_descriptors(im_gray: np.ndarray,
                      coords: np.ndarray,
                      win: int = 40,
                      blur_sigma: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        im_gray: HxW grayscale float32 in [0,1]
        coords:  2xN array (ys, xs)
        win:     context window (40)
        blur_sigma: Gaussian sigma applied on the 40x40 patch before pooling

    Returns:
        descs:  N x 64 array (each row is an 8x8 normalized descriptor)
        kept:   2 x N array of coords for which a descriptor was produced
    """
    if coords.size == 0:
        return np.zeros((0, 64), dtype=np.float32), coords

    ys, xs = coords
    descs = []
    kept_y, kept_x = [], []

    for y, x in zip(ys, xs):
        p40 = extract_patch_40(im_gray, int(y), int(x), win=win)
        if blur_sigma and blur_sigma > 0:
            p40 = gaussian_blur(p40, blur_sigma)
        d8 = downsample_40_to_8(p40)
        v  = normalize_descriptor(d8)
        descs.append(v)
        kept_y.append(y); kept_x.append(x)

    descs = np.stack(descs, axis=0) if descs else np.zeros((0, 64), dtype=np.float32)
    kept  = np.stack([np.array(kept_y, dtype=int), np.array(kept_x, dtype=int)], axis=0) if descs.size else coords
    return descs, kept

def show_example_descriptors(im_gray: np.ndarray, coords: np.ndarray,
                             descs: np.ndarray, k: int = 6):
    """
    Displays for each selected keypoint:
    - the full image with the keypoint marked
    - a zoomed-in 40x40 patch centered at the keypoint
    - its corresponding normalized 8x8 descriptor
    """
    k = min(k, descs.shape[0])
    fig, axes = plt.subplots(3, k, figsize=(2*k, 6))
    if k == 1:
        axes = np.expand_dims(axes, axis=1)

    for i in range(k):
        y, x = coords[0, i], coords[1, i]

        # ---- Full image with keypoint
        axes[0, i].imshow(im_gray, cmap='gray')
        axes[0, i].scatter([x], [y], s=40, facecolors='none', edgecolors='lime', linewidths=1.5)
        axes[0, i].set_title(f'Keypoint {i}', fontsize=10)
        axes[0, i].axis('off')

        # ---- Zoomed 40x40 patch
        half = 20
        H, W = im_gray.shape
        y0, y1 = max(0, y-half), min(H, y+half)
        x0, x1 = max(0, x-half), min(W, x+half)
        patch = im_gray[y0:y1, x0:x1]
        axes[1, i].imshow(patch, cmap='gray')
        axes[1, i].set_title('40×40 patch', fontsize=9)
        axes[1, i].axis('off')

        # ---- 8×8 descriptor (reshaped from vector)
        axes[2, i].imshow(descs[i].reshape(8, 8), cmap='gray', interpolation='nearest')
        axes[2, i].set_title('8×8 descriptor', fontsize=9)
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()


# --------------- main -------------------
if __name__ == "__main__":
    image_path = "./media/park.jpg"  # change if needed
    num_points = 200                   # just to visualize fewer

    TO_GRAYSCALE = HD.to_grayscale
    GET_CORNERS = getattr(HD, "get_harris_corners_numpy", None) or HD.get_harris_corners

    print(f"Loading: {image_path}")
    img = plt.imread(image_path)
    img_gray = TO_GRAYSCALE(img)

    # Grab keypoints from your B.1 implementation (numpy Harris or skimage fallback)
    print("Detecting keypoints...")
    h, coords = GET_CORNERS(img_gray, edge_discard=20, sigma=1.5, min_distance=1, threshold_rel=0.01)
    # keep the first N just for demo display
    if coords.shape[1] > num_points:
        coords = coords[:, :num_points]

    print("Building descriptors (40×40 → blur → 8×8 → normalize)...")
    descs, kept = build_descriptors(img_gray, coords, win=40, blur_sigma=2.0)
    print("Descriptors shape:", descs.shape)   # (N, 64)

    # Show a few
    show_example_descriptors(img_gray, kept, descs, k=6)
