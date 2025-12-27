import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import corner_harris, peak_local_max
from skimage.color import rgb2gray, rgba2rgb


# -------------- Harris + peak picking -----------------------------
def get_harris_corners(im, edge_discard=20, sigma=1.5, min_distance=1, threshold_rel=0.01):
    """
    This function takes a b&w image and an optional amount to discard
    on the edge (default is 5 pixels), and finds all harris corners
    in the image. Harris corners near the edge are discarded and the
    coordinates of the remaining corners are returned. A 2d array (h)
    containing the h value of every pixel is also returned.

    h is the same shape as the original image, im.
    coords is 2 x n (ys, xs).
    """
    assert edge_discard >= 20, "Use at least 20px per the assignment."

    # 1) Harris response
    h = corner_harris(im, method='eps', sigma=sigma)

    # 2) Local maxima of h (corner candidates)
    # NOTE: 'indices' kwarg is removed in skimage>=0.20 — just omit it.
    peaks = peak_local_max(
        h,
        min_distance=min_distance,
        threshold_rel=threshold_rel
    )  # shape: (N, 2) -> rows=y, cols=x

    if peaks.size == 0:
        return h, np.zeros((2, 0), dtype=int)

    # 3) Discard peaks near edges
    r, c = peaks[:, 0], peaks[:, 1]
    H, W = im.shape[:2]
    mask = (r > edge_discard) & (r < H - edge_discard) & (c > edge_discard) & (c < W - edge_discard)
    coords = np.stack([r[mask], c[mask]], axis=0)  # shape: 2 x N

    return h, coords


# --------------- Squared distance helper -----------------------------
def dist2(x, c):
    """
    dist2  Calculates squared distance between two sets of points.

    Description
    D = DIST2(X, C) takes two matrices of vectors and calculates the
    squared Euclidean distance between them.  Both matrices must be of
    the same column dimension.  If X has M rows and N columns, and C has
    L rows and N columns, then the result has M rows and L columns.  The
    I, Jth entry is the  squared distance from the Ith row of X to the
    Jth row of C.

    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """
    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert dimx == dimc, 'Data dimension does not match dimension of centers'

    return (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T + \
           np.ones((   ndata, 1)) * np.sum((c**2).T, axis=0)    - \
           2 * np.inner(x, c)


# --------------- ANMS (Brown et al. style) -----------------------------
def anms(h, coords, num_keep=500, c_robust=0.9):
    """
    Adaptive Non-Maximal Suppression.
    """
    if coords.size == 0:
        return coords, np.array([])

    ys, xs = coords
    vals = h[ys, xs]  # corner strengths
    pts = np.stack([ys, xs], axis=1).astype(np.float64)  # N x 2

    N = pts.shape[0]
    D2 = dist2(pts, pts)  # N x N
    np.fill_diagonal(D2, np.inf)

    radii = np.full(N, np.inf, dtype=np.float64)
    for i in range(N):
        stronger = np.where(vals > c_robust * vals[i])[0]
        if stronger.size > 0:
            radii[i] = np.min(D2[i, stronger])

    order = np.argsort(-radii)  # descending
    take = order[:min(num_keep, N)]
    coords_kept = coords[:, take]
    return coords_kept, radii[take]


# ------------------ Visualization helpers -----------------------------
def show_corners(im, coords, title="Corners", s=12):
    """
    im: grayscale or RGB image array
    coords: 2 x N (ys, xs)
    """
    plt.figure(figsize=(6, 6))
    if im.ndim == 2:
        plt.imshow(im, cmap='gray')
    else:
        plt.imshow(im)
    if coords.size:
        ys, xs = coords
        # Add edgecolors for visibility and to avoid warnings
        plt.scatter(xs, ys, s=s, marker='o', facecolors='none', edgecolors='lime')
    plt.title(title)
    plt.axis('off')
    plt.show()

def to_grayscale(img):
    """
    Robustly convert to grayscale float32 in [0,1].
    Handles 2D, RGB, and RGBA inputs.
    """
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0

    if img.ndim == 2:
        g = img  # already grayscale
    elif img.ndim == 3 and img.shape[2] == 3:
        g = rgb2gray(img)
    elif img.ndim == 3 and img.shape[2] == 4:
        # Compose RGBA over white background, then gray
        rgb = rgba2rgb(img)     # float in [0,1]
        g = rgb2gray(rgb)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    return g.astype(np.float32)


if __name__ == "__main__":
    # Hardcoded parameters
    image_path = "./media/park.jpg"   # <-- change this path as needed
    num_points = 500                    # number of corners to keep after ANMS
    sigma = 1.5
    threshold_rel = 0.01
    min_distance = 1
    edge_discard = 20

    print(f"Loading image: {image_path}")
    img = plt.imread(image_path)
    img_gray = to_grayscale(img)

    print("Computing Harris corners...")
    h, coords = get_harris_corners(
        img_gray,
        edge_discard=edge_discard,
        sigma=sigma,
        min_distance=min_distance,
        threshold_rel=threshold_rel
    )
    print(f"Detected {coords.shape[1]} corners before ANMS")

    show_corners(img_gray, coords, title=f"Harris Corners (No ANMS) — {coords.shape[1]} pts")

    print("Applying ANMS...")
    coords_anms, _ = anms(h, coords, num_keep=num_points, c_robust=0.9)
    print(f"Kept {coords_anms.shape[1]} corners after ANMS")

    show_corners(img_gray, coords_anms, title=f"Harris Corners (ANMS) — {coords_anms.shape[1]} pts")
