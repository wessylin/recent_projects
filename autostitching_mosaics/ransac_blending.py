import numpy as np
import matplotlib.pyplot as plt

import harris_detection as HD            # B.1: to_grayscale, corners, ANMS
import feature_extraction as FE          # B.2: build_descriptors
from main_blend import (                 # Part A helpers 
    compute_union_canvas,
    warp_source_to_union,
    place_reference_on_union,
    feather_blend_union,
    crop_to_ref_window,
    show_four
)

# -------------------- Matching (same logic you used in B.3) --------------------
def cdist_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.sum(a*a, axis=1, keepdims=True)
    b2 = np.sum(b*b, axis=1, keepdims=True).T
    return a2 + b2 - 2*np.dot(a, b.T)

def match_descriptors(descA: np.ndarray, descB: np.ndarray, ratio_thresh: float = 0.75) -> np.ndarray:
    if descA.size == 0 or descB.size == 0:
        return np.zeros((0, 2), dtype=int)
    D2  = cdist_l2(descA, descB)
    idx1 = np.argpartition(D2, 1, axis=1)[:, 0]
    d1   = D2[np.arange(D2.shape[0]), idx1]
    D2m  = D2.copy()
    D2m[np.arange(D2.shape[0]), idx1] = np.inf
    idx2 = np.argpartition(D2m, 1, axis=1)[:, 0]
    d2   = D2m[np.arange(D2.shape[0]), idx2]
    keep = (d2 > 0) & ((d1 / d2) < (ratio_thresh**2))
    return np.stack([np.arange(D2.shape[0])[keep], idx1[keep]], axis=1)

def filter_keypoints_by_overlap(coordsA, coordsB, W1, W2, overlap_frac=0.5):
    # ref is LEFT, tgt is RIGHT
    ysA, xsA = coordsA; ysB, xsB = coordsB
    xA_min = int(W1 * (1.0 - overlap_frac))
    xB_max = int(W2 * overlap_frac)
    maskA = xsA >= xA_min
    maskB = xsB <= xB_max
    return coordsA[:, maskA], coordsB[:, maskB]

# -------------------- DLT Homography (normalized) + 4pt RANSAC -----------------
def _normalize_points(pts):
    m = pts.mean(axis=0)
    s = (pts - m).std(axis=0).mean() + 1e-8
    a = np.sqrt(2.0) / s
    T = np.array([[a,0,-a*m[0]],[0,a,-a*m[1]],[0,0,1]], dtype=np.float64)
    P = (T @ np.c_[pts, np.ones(len(pts))].T).T
    return T, P[:, :2]

def dlt_homography(src, dst):
    T1, s = _normalize_points(src)
    T2, d = _normalize_points(dst)
    A = []
    for (x,y),(u,v) in zip(s, d):
        A.append([ 0,0,0, -x,-y,-1, v*x, v*y, v])
        A.append([ x,y,1,  0, 0, 0,-u*x,-u*y,-u])
    A = np.asarray(A)
    _, _, VT = np.linalg.svd(A)
    h = VT[-1] / (VT[-1,-1] + 1e-12)
    Hn = h.reshape(3,3)
    H  = np.linalg.inv(T2) @ Hn @ T1
    return H / (H[2,2] + 1e-12)

def ransac_homography(pts_src, pts_ref, num_iter=2000, reproj_thresh=3.0, seed=42):
    """
    Estimate H s.t. ref ~ H * src. Returns (H, inlier_mask).
    """
    rng = np.random.default_rng(seed)
    N = pts_src.shape[0]
    if N < 4: return None, np.zeros(N, bool)

    A_h = np.c_[pts_src, np.ones(N)]
    best_cnt, best_in, best_H = 0, np.zeros(N, bool), None

    for _ in range(num_iter):
        idx = rng.choice(N, 4, replace=False)
        H = dlt_homography(pts_src[idx], pts_ref[idx])
        proj = (H @ A_h.T).T
        proj = proj[:, :2] / (proj[:, 2:3] + 1e-12)
        err  = np.linalg.norm(proj - pts_ref, axis=1)
        inl  = err < reproj_thresh
        cnt  = inl.sum()
        if cnt > best_cnt:
            best_cnt, best_in, best_H = cnt, inl, H

    if best_cnt >= 4:
        best_H = dlt_homography(pts_src[best_in], pts_ref[best_in])
    return best_H, best_in

# ----------------------------- Visualization (inliers) ---------------------------
def draw_inliers(img_ref, img_tgt, coords_ref, coords_tgt, pairs, inliers):
    H1, W1 = img_ref.shape[:2]; H2, W2 = img_tgt.shape[:2]
    Hc = max(H1, H2)
    canvas = np.ones((Hc, W1 + W2, 3), dtype=float)
    A = img_ref.astype(float); B = img_tgt.astype(float)
    if A.max() > 1.0: A /= 255.0
    if B.max() > 1.0: B /= 255.0
    if A.ndim == 2: A = np.stack([A, A, A], axis=2)
    if B.ndim == 2: B = np.stack([B, B, B], axis=2)
    canvas[:H1, :W1] = A; canvas[:H2, W1:W1+W2] = B

    pr = pairs[inliers]
    plt.figure(figsize=(10,6)); plt.imshow(canvas); 
    for iA, jB in pr:
        y1, x1 = coords_ref[0, iA], coords_ref[1, iA]
        y2, x2 = coords_tgt[0, jB], coords_tgt[1, jB]
        plt.plot([x1, x2 + W1], [y1, y2], '-', linewidth=0.8)
    plt.title(f"Inlier matches ({pr.shape[0]})"); plt.axis('off'); plt.tight_layout(); plt.show()

# --------------------------------------- Main ------------------------------------
if __name__ == "__main__":
    ref_path = "./media/room_1.jpg"
    tgt_path = "./media/room_2.jpg"

    # Overlap/matcher/detector/RANSAC knobs
    overlap_frac  = 0.7
    ratio_thresh  = 0.75
    edge_discard  = 20
    sigma         = 1.5
    threshold_rel = 0.01
    min_distance  = 1
    anms_keep     = 80
    ransac_iters  = 2000
    reproj_thr    = 3.0

    # Load + grayscale for detection/desc; keep originals for final mosaic display
    ref_rgb = plt.imread(ref_path)
    tgt_rgb = plt.imread(tgt_path)
    ref_gray = HD.to_grayscale(ref_rgb)
    tgt_gray = HD.to_grayscale(tgt_rgb)

    H1, W1 = ref_gray.shape; H2, W2 = tgt_gray.shape

    # --- B.1 corners (+ANMS)
    get_corners = getattr(HD, "get_harris_corners_numpy", None) or HD.get_harris_corners
    hA, cA = get_corners(ref_gray, edge_discard=edge_discard, sigma=sigma,
                         min_distance=min_distance, threshold_rel=threshold_rel)
    hB, cB = get_corners(tgt_gray, edge_discard=edge_discard, sigma=sigma,
                         min_distance=min_distance, threshold_rel=threshold_rel)
    if hasattr(HD, "anms"):
        cA, _ = HD.anms(hA, cA, num_keep=anms_keep, c_robust=0.9)
        cB, _ = HD.anms(hB, cB, num_keep=anms_keep, c_robust=0.9)

    # --- Overlap gating (left/right)
    cA_f, cB_f = filter_keypoints_by_overlap(cA, cB, W1, W2, overlap_frac=overlap_frac)

    # --- B.2 descriptors
    dA, cA_f = FE.build_descriptors(ref_gray, cA_f, win=40, blur_sigma=2.0)
    dB, cB_f = FE.build_descriptors(tgt_gray, cB_f, win=40, blur_sigma=2.0)

    # --- B.3 matching (Lowe)
    pairs = match_descriptors(dA, dB, ratio_thresh=ratio_thresh)
    if pairs.shape[0] == 0:
        print("No matches after ratio test."); raise SystemExit

    # coords → (x,y)
    pts_ref = np.stack([cA_f[1, pairs[:,0]], cA_f[0, pairs[:,0]]], axis=1)
    pts_tgt = np.stack([cB_f[1, pairs[:,1]], cB_f[0, pairs[:,1]]], axis=1)

    # --- B.4 RANSAC: estimate H mapping TARGET -> REFERENCE
    H, inliers = ransac_homography(pts_ref,pts_tgt, num_iter=ransac_iters, reproj_thresh=reproj_thr)

    print(f"Inliers: {inliers.sum()} / {len(inliers)}")

    # Optional: visualize inliers
    draw_inliers(ref_gray, tgt_gray, cA_f, cB_f, pairs, inliers)

    # --- Use your Part A helpers to warp & blend on a union canvas
    union_shape, ref_offset = compute_union_canvas(src=ref_rgb, ref=tgt_rgb, H_src_to_ref=H)
    warped_union, validB    = warp_source_to_union(src=ref_rgb, H_src_to_ref=H,
                                                   union_shape=union_shape, ref_offset=ref_offset)
    A_union, mA             = place_reference_on_union(ref=tgt_rgb, union_shape=union_shape, ref_offset=ref_offset)
    blended_uncropped       = feather_blend_union(A_union, warped_union, mA, validB, sigma=25.0)
    blended_cropped         = crop_to_ref_window(blended_uncropped, ref_offset, (ref_rgb.shape[0], ref_rgb.shape[1]))

    # 4-up for the report
    show_four(tgt_rgb, ref_rgb, blended_uncropped, blended_cropped)
