import numpy as np
import matplotlib.pyplot as plt
from main_homography import computeH   
from main_warp import click_points, _warp_Image_Bilinear

# ---------------------------- Image handling --------------------------

def _to_rgb3(im: np.ndarray) -> np.ndarray:
    """Return an RGB (H,W,3) view of im. Grayscale is stacked; RGBA drops alpha."""
    if im.ndim == 2:                            # grayscale → RGB
        return np.stack([im, im, im], axis=2)
    if im.ndim == 3:
        if im.shape[2] == 3:                    # already RGB
            return im
        if im.shape[2] >= 4:                    # RGBA or more → take first 3
            return im[..., :3]
        
# ----------------------------- Gaussian blur --------------------------

def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 0: return np.array([1.0], dtype=float)
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-(x * x) / (2 * sigma * sigma)); k /= k.sum()
    return k

def _convolve1d_reflect(arr: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    pad = len(k) // 2
    pads = [(0,0)] * arr.ndim
    pads[axis] = (pad, pad)
    arr_pad = np.pad(arr, pads, mode="reflect")
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), axis, arr_pad)

def gaussian_blur2d(img: np.ndarray, sigma: float) -> np.ndarray:
    k = _gaussian_kernel1d(sigma)
    if img.ndim == 2:
        tmp = _convolve1d_reflect(img.astype(float), k, axis=1)
        return _convolve1d_reflect(tmp, k, axis=0)
    out = np.empty_like(img, dtype=float)
    for c in range(img.shape[2]):
        tmp = _convolve1d_reflect(img[..., c].astype(float), k, axis=1)
        out[..., c] = _convolve1d_reflect(tmp, k, axis=0)
    return out

# ----------------------------- Union canvas  -----------------------------

def _project_corners(Ws: int, Hs: int, H: np.ndarray) -> np.ndarray:
    corners = np.array([[0,0,1],[Ws-1,0,1],[Ws-1,Hs-1,1],[0,Hs-1,1]], dtype=float).T  # (3,4)
    Q = H @ corners
    Q /= Q[2:3, :]
    return Q[:2].T  

def compute_union_canvas(src: np.ndarray, ref: np.ndarray, H_src_to_ref: np.ndarray):
    Hs, Ws = src.shape[0], src.shape[1]
    Hr, Wr = ref.shape[0], ref.shape[1]
    pts = _project_corners(Ws, Hs, H_src_to_ref)  # in ref coords
    x_min = int(np.floor(min(0, pts[:,0].min())))
    y_min = int(np.floor(min(0, pts[:,1].min())))
    x_max = int(np.ceil(max(Wr-1, pts[:,0].max())))
    y_max = int(np.ceil(max(Hr-1, pts[:,1].max())))
    W_u = x_max - x_min + 1
    H_u = y_max - y_min + 1
    off_x, off_y = -x_min, -y_min  # place reference TL at (off_x, off_y)
    return (H_u, W_u), (off_x, off_y)

# ------------------------------ Warp to union canvas ------------------------------
def warp_source_to_union(src: np.ndarray, H_src_to_ref: np.ndarray,
                         union_shape: tuple[int,int], ref_offset: tuple[int,int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Warp SOURCE into the union canvas by inverse mapping via H^{-1} in *reference* coordinates.
    """
    H_u, W_u = union_shape
    off_x, off_y = ref_offset
    src3 = _to_rgb3(src).astype(np.float32)
    if src3.max() > 1.0: src3 /= 255.0
    H_inv = np.linalg.inv(H_src_to_ref)

    # union grid -> reference coordinates
    Xu, Yu = np.meshgrid(np.arange(W_u), np.arange(H_u))
    Xr = Xu - off_x
    Yr = Yu - off_y

    x_s, y_s = _inverse_map_to_source(H_inv, Xr, Yr)
    warped, valid = _warp_Image_Bilinear(src3, x_s, y_s)  # (H_u, W_u, 3), (H_u, W_u)
    return warped, valid

def place_reference_on_union(ref: np.ndarray, union_shape: tuple[int,int], ref_offset: tuple[int,int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Paste REFERENCE into the union canvas at (off_x, off_y). Also return its hard mask.
    """
    H_u, W_u = union_shape
    off_x, off_y = ref_offset
    ref3 = _to_rgb3(ref).astype(np.float32)
    if ref3.max() > 1.0: ref3 /= 255.0

    A = np.zeros((H_u, W_u, 3), dtype=np.float32)
    mA = np.zeros((H_u, W_u), dtype=np.float32)
    Hr, Wr = ref3.shape[0], ref3.shape[1]
    A[off_y:off_y+Hr, off_x:off_x+Wr, :] = ref3
    mA[off_y:off_y+Hr, off_x:off_x+Wr] = 1.0
    return A, mA

# ------------------------------- Feather blending ---------------------------------

def feather_blend_union(A: np.ndarray, B: np.ndarray, mA: np.ndarray, mB: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """
    Soft-mask weighted averaging on the union canvas:
      wA = Gσ(mA),  wB = Gσ(mB),  out = (wA·A + wB·B) / (wA + wB + ε)
    """
    wA = gaussian_blur2d(mA, sigma=sigma)[..., None]
    wB = gaussian_blur2d(mB, sigma=sigma)[..., None]
    W = wA + wB + 1e-8
    out = (wA * A + wB * B) / W
    return np.clip(out, 0.0, 1.0)

# ---------------------------------- Cropping --------------------------------------

def crop_to_ref_window(img_union: np.ndarray, ref_offset: tuple[int,int], ref_shape: tuple[int,int]) -> np.ndarray:
    off_x, off_y = ref_offset
    Hr, Wr = ref_shape
    return img_union[off_y:off_y+Hr, off_x:off_x+Wr, :]

# ---------------------------------- Visualizer ------------------------------------

def show_four(src_img, ref_img, blended_uncropped, blended_cropped):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    axes[0].imshow(src_img);             axes[0].set_title("Source (to warp)");           axes[0].axis("off")
    axes[1].imshow(ref_img);             axes[1].set_title("Reference (target)");         axes[1].axis("off")
    axes[2].imshow(blended_uncropped);   axes[2].set_title("Blended (uncropped union)");  axes[2].axis("off")
    axes[3].imshow(blended_cropped);     axes[3].set_title("Blended (cropped = ref)");    axes[3].axis("off")
    plt.tight_layout(); plt.show()

# ------------------------------------- Main ---------------------------------------

def main():
    
    A_path = "./media/park_1.jpg"   # source to warp
    B_path = "./media/park_2.jpg"   # reference frame

    src = plt.imread(A_path); ref = plt.imread(B_path)
    if src.ndim == 3 and src.shape[2] == 4: src = src[..., :3]
    if ref.ndim == 3 and ref.shape[2] == 4: ref = ref[..., :3]

    n = 8
    print(f"SOURCE image:")
    P_src = click_points(src, n, "SOURCE")
    print(f"REFERENCE image:")
    P_ref = click_points(ref, n, "REFERENCE")

    # H: source -> reference (A.2)
    H = computeH(P_src, P_ref)
    print("\n=== Homography SOURCE → REFERENCE (scaled so H[2,2]≈1) ===")
    with np.printoptions(precision=6, suppress=True): print(H)

    # Union canvas (uncropped)
    union_shape, ref_offset = compute_union_canvas(src, ref, H)
    warped_union, validB = warp_source_to_union(src, H, union_shape, ref_offset)
    A_union, mA = place_reference_on_union(ref, union_shape, ref_offset)

    # Feather blend on union
    blended_uncropped = feather_blend_union(A_union, warped_union, mA, validB, sigma=25.0)

    # Cropped = reference window extracted from union blend
    blended_cropped = crop_to_ref_window(blended_uncropped, ref_offset, (ref.shape[0], ref.shape[1]))

    # 4-up view
    show_four(src, ref, blended_uncropped, blended_cropped)

if __name__ == "__main__":
    main()
