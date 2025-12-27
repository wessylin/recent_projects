from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from main_homography import computeH, click_points

# ------------------------------ Warp core ------------------------------------

def _ensure_3ch(im: np.ndarray) -> tuple[np.ndarray, bool]:
    if im.ndim == 2:
        return np.stack([im, im, im], axis=2), True
    return im, False

def _inverse_map_to_source(H_inv: np.ndarray, W_ref: int, H_ref: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a grid over the reference frame (0..W_ref-1, 0..H_ref-1), then inverse-map
    through H_inv to get source coordinates (x_s, y_s) for every reference pixel.
    """
    xs = np.arange(W_ref)
    ys = np.arange(H_ref)
    X, Y = np.meshgrid(xs, ys)  # ref coords
    ones = np.ones_like(X, dtype=float)
    P = np.stack([X, Y, ones], axis=-1).reshape(-1, 3).T  # (3, N)
    Q = H_inv @ P
    x_s = (Q[0] / Q[2]).reshape(H_ref, W_ref)
    y_s = (Q[1] / Q[2]).reshape(H_ref, W_ref)
    return x_s, y_s

def _warp_Image_Nearest_Neighbor(im3: np.ndarray, x_s: np.ndarray, y_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest neighbor sampling into reference-sized grid. Returns (warped, valid_mask)."""
    Hs, Ws, _ = im3.shape
    xi = np.rint(x_s).astype(int)
    yi = np.rint(y_s).astype(int)
    valid = (xi >= 0) & (xi < Ws) & (yi >= 0) & (yi < Hs)
    out = np.zeros((y_s.shape[0], x_s.shape[1], 3), dtype=im3.dtype)
    for c in range(3):
        tmp = out[..., c]
        tmp[valid] = im3[yi[valid], xi[valid], c]
        out[..., c] = tmp
    return out, valid

def _warp_Image_Bilinear(im3: np.ndarray, x_s: np.ndarray, y_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear sampling into reference-sized grid. Returns (warped, valid_mask)."""
    Hs, Ws, _ = im3.shape
    x0 = np.floor(x_s).astype(int); x1 = x0 + 1
    y0 = np.floor(y_s).astype(int); y1 = y0 + 1
    valid = (x0 >= 0) & (x1 < Ws) & (y0 >= 0) & (y1 < Hs)

    ax = (x_s - x0).clip(0.0, 1.0)
    ay = (y_s - y0).clip(0.0, 1.0)

    out = np.zeros((y_s.shape[0], x_s.shape[1], 3), dtype=float)
    for c in range(3):
        I = im3[..., c]
        Ia = np.zeros_like(x_s, dtype=float); Ib = np.zeros_like(x_s, dtype=float)
        Ic = np.zeros_like(x_s, dtype=float); Id = np.zeros_like(x_s, dtype=float)
        Ia[valid] = I[y0[valid], x0[valid]]  # TL
        Ib[valid] = I[y0[valid], x1[valid]]  # TR
        Ic[valid] = I[y1[valid], x0[valid]]  # BL
        Id[valid] = I[y1[valid], x1[valid]]  # BR
        out[..., c] = (
            (1 - ax) * (1 - ay) * Ia +
            ax       * (1 - ay) * Ib +
            (1 - ax) * ay       * Ic +
            ax       * ay       * Id
        )

    # cast back to input dtype range
    if np.issubdtype(im3.dtype, np.integer):
        out = np.clip(out, 0, 255).astype(im3.dtype)
    else:
        out = np.clip(out, 0.0, 1.0).astype(im3.dtype)
    return out, valid

def warp_to_reference(im_src: np.ndarray, H_src_to_ref: np.ndarray, ref_shape: tuple[int, int], method: str) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-warp SOURCE into the REFERENCE frame size."""
    
    assert method in {"nearest", "bilinear"}
    im3, was_gray = _ensure_3ch(im_src)
    H_ref, W_ref = ref_shape
    H_inv = np.linalg.inv(H_src_to_ref)

    x_s, y_s = _inverse_map_to_source(H_inv, W_ref, H_ref)

    if method == "nearest":
        warped3, valid = _warp_Image_Nearest_Neighbor(im3, x_s, y_s)
    else:
        warped3, valid = _warp_Image_Bilinear(im3, x_s, y_s)

    if was_gray:
        warped = warped3[..., 0]
    else:
        warped = warped3
    return warped, valid

# ----------------------------------- Vizualizer -----------------------------------

def show_visualizer(ref_img, warped_nn, warped_bil):
    fig, axes = plt.subplots(1, 3, figsize=(10, 8))
    axes = axes.ravel()
    axes[0].imshow(ref_img);        axes[0].set_title("Reference");          axes[0].axis("off")
    axes[1].imshow(warped_nn);      axes[1].set_title("Warped (Nearest)");   axes[1].axis("off")
    axes[2].imshow(warped_bil);     axes[2].set_title("Warped (Bilinear)");  axes[2].axis("off")
    plt.tight_layout()
    plt.show()

# ---------------------------------- Main -----------------------------------
def main():

    A_path = "./media/regent_1.jpg"
    B_path = "./media/regent_2.jpg"

    src_img = plt.imread(A_path)
    ref_img = plt.imread(B_path)

    num_points = 8

    # 1. Select correspondences
    print(f"Click {num_points} correspondences on SOURCE image:")
    pts_src = click_points(src_img, num_points, title="SOURCE")
    print(f"Click {num_points} correspondences on REFERENCE image:")
    pts_ref = click_points(ref_img, num_points, title="REFERENCE")

    # 2. Compute H (source -> reference)
    H = computeH(pts_src, pts_ref)
    print("\n=== Homography SOURCE → REFERENCE ===")
    with np.printoptions(precision=6, suppress=True):
        print(H)

    # 3. Warp SOURCE into REFERENCE frame
    H_ref, W_ref = ref_img.shape[0], ref_img.shape[1]
    warped_nn, valid_nn   = warp_to_reference(src_img, H, (H_ref, W_ref), method="nearest")
    warped_bi, valid_bi   = warp_to_reference(src_img, H, (H_ref, W_ref), method="bilinear")

    # 4. Show visualizer: Reference, Warped (NN),Warped (Bilinear)
    show_visualizer(ref_img, warped_nn, warped_bi)

if __name__ == "__main__":
    main()
