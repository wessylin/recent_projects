import numpy as np
import matplotlib.pyplot as plt
import harris_detection as HD           # B.1
import feature_extraction as FE         # B.2

# ---------- L2 distances ----------
def cdist_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.sum(a*a, axis=1, keepdims=True)
    b2 = np.sum(b*b, axis=1, keepdims=True).T
    return a2 + b2 - 2*np.dot(a, b.T)

# ---------- Lowe ratio test ----------
def match_descriptors(descA: np.ndarray, descB: np.ndarray, ratio_thresh: float = 0.75) -> np.ndarray:
    if descA.size == 0 or descB.size == 0:
        return np.zeros((0, 2), dtype=int)
    D2 = cdist_l2(descA, descB)
    idx1 = np.argpartition(D2, 1, axis=1)[:, 0]
    d1 = D2[np.arange(D2.shape[0]), idx1]
    D2m = D2.copy()
    D2m[np.arange(D2.shape[0]), idx1] = np.inf
    idx2 = np.argpartition(D2m, 1, axis=1)[:, 0]
    d2 = D2m[np.arange(D2.shape[0]), idx2]
    keep = (d2 > 0) & ((d1 / d2) < (ratio_thresh**2))
    return np.stack([np.arange(D2.shape[0])[keep], idx1[keep]], axis=1)

# ---------- Overlap filters (no direction/y_tol) ----------
def filter_keypoints_by_overlap(coordsA, coordsB, W1, W2, overlap_frac=0.5):
    """
    ref is LEFT, tgt is RIGHT.
    Keep: ref x >= W1*(1-overlap), tgt x <= W2*overlap.
    Return filtered coords and index maps to original arrays.
    """
    ysA, xsA = coordsA
    ysB, xsB = coordsB
    xA_min = int(W1 * (1.0 - overlap_frac))
    xB_max = int(W2 * overlap_frac)
    maskA = xsA >= xA_min
    maskB = xsB <= xB_max
    coordsA_f = coordsA[:, maskA]
    coordsB_f = coordsB[:, maskB]
    idxA = np.where(maskA)[0]
    idxB = np.where(maskB)[0]
    return coordsA_f, coordsB_f, idxA, idxB

def filter_matches_overlap_only(pairs, coordsA_f, coordsB_f, W1, W2, overlap_frac=0.5):
    """
    Enforce x-positions to be inside their overlap strips (redundant but safe).
    No vertical constraint.
    """
    if pairs.size == 0:
        return pairs
    ysA, xsA = coordsA_f
    ysB, xsB = coordsB_f
    xA_min = int(W1 * (1.0 - overlap_frac))
    xB_max = int(W2 * overlap_frac)
    keep = []
    for iA, jB in pairs:
        if xsA[iA] >= xA_min and xsB[jB] <= xB_max:
            keep.append((iA, jB))
    return np.array(keep, dtype=int) if keep else np.zeros((0, 2), dtype=int)

# ---------- Visualization ----------
def draw_matches(imgA, imgB, coordsA, coordsB, matches, max_draw=120):
    H1, W1 = imgA.shape
    H2, W2 = imgB.shape
    H = max(H1, H2)
    canvas = np.ones((H, W1 + W2), dtype=imgA.dtype)
    canvas[:H1, :W1] = imgA
    canvas[:H2, W1:W1+W2] = imgB

    plt.figure(figsize=(10, 6))
    plt.imshow(canvas, cmap='gray')

    if matches.shape[0] > max_draw:
        matches = matches[:max_draw]

    for iA, jB in matches:
        y1, x1 = coordsA[0, iA], coordsA[1, iA]
        y2, x2 = coordsB[0, jB], coordsB[1, jB]
        plt.plot([x1, x2 + W1], [y1, y2], '-', linewidth=0.9)
        plt.scatter([x1, x2 + W1], [y1, y2], s=15, facecolors='none')
    plt.title(f"Matched features (shown {matches.shape[0]})")
    plt.axis('off'); plt.tight_layout(); plt.show()

# ---------- MAIN ----------
if __name__ == "__main__":
    # Hardcoded paths: left/ref and right/tgt
    ref_path = "./media/park_1.jpg"
    tgt_path = "./media/park_2.jpg"

    overlap_frac = 0.5       # ~50% horizontal overlap
    edge_discard = 20
    sigma = 1.5
    threshold_rel = 0.01
    min_distance = 1
    anms_keep = 500
    ratio_thresh = 0.75
    max_draw = 20

    # Load + grayscale
    ref_rgb = plt.imread(ref_path)
    tgt_rgb = plt.imread(tgt_path)
    ref = HD.to_grayscale(ref_rgb)
    tgt = HD.to_grayscale(tgt_rgb)
    H1, W1 = ref.shape
    H2, W2 = tgt.shape

    # Keypoints (B.1)
    get_corners = getattr(HD, "get_harris_corners_numpy", None) or HD.get_harris_corners
    hA, coordsA = get_corners(ref, edge_discard=edge_discard, sigma=sigma,
                              min_distance=min_distance, threshold_rel=threshold_rel)
    hB, coordsB = get_corners(tgt, edge_discard=edge_discard, sigma=sigma,
                              min_distance=min_distance, threshold_rel=threshold_rel)
    if hasattr(HD, "anms"):
        coordsA, _ = HD.anms(hA, coordsA, num_keep=anms_keep, c_robust=0.9)
        coordsB, _ = HD.anms(hB, coordsB, num_keep=anms_keep, c_robust=0.9)

    # --- Overlap gating (pre-descriptor) ---
    coordsA_f, coordsB_f, idxA, idxB = filter_keypoints_by_overlap(coordsA, coordsB, W1, W2, overlap_frac)

    # Descriptors (B.2)
    descA, coordsA_f = FE.build_descriptors(ref, coordsA_f, win=40, blur_sigma=2.0)
    descB, coordsB_f = FE.build_descriptors(tgt, coordsB_f, win=40, blur_sigma=2.0)

    # Matching
    pairs = match_descriptors(descA, descB, ratio_thresh=ratio_thresh)

    # --- Overlap enforcement (post) ---
    pairs = filter_matches_overlap_only(pairs, coordsA_f, coordsB_f, W1, W2, overlap_frac)

    # Sort by best distance to draw “best first”
    if pairs.shape[0] > 0:
        D2 = cdist_l2(descA[pairs[:, 0]], descB)
        d1 = D2[np.arange(pairs.shape[0]), pairs[:, 1]]
        order = np.argsort(d1)
        pairs = pairs[order]

    print(f"Keypoints after ANMS: A={coordsA.shape[1]}, B={coordsB.shape[1]}")
    print(f"After overlap gating:  A={coordsA_f.shape[1]}, B={coordsB_f.shape[1]}")
    print(f"Matches kept (ratio<{ratio_thresh}, overlap={int(overlap_frac*100)}%): {pairs.shape[0]}")

    draw_matches(ref, tgt, coordsA_f, coordsB_f, pairs, max_draw=max_draw)
