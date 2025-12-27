import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt

# ---------------------- DLT for homography ----------------------
def build_A(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
      [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
      [ 0,  0,  0,-x,-y,-1, v*x, v*y, v]
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    assert src.shape == dst.shape and src.ndim == 2 and src.shape[1] == 2, "src/dst must be (N,2)"
    N = src.shape[0]
    assert N >= 4, "Need at least 4 correspondences"

    x, y = src[:, 0], src[:, 1]
    u, v = dst[:, 0], dst[:, 1]

    A = np.zeros((2 * N, 9), dtype=float)
    A[0::2, 0] = -x
    A[0::2, 1] = -y
    A[0::2, 2] = -1
    A[0::2, 6] = u * x
    A[0::2, 7] = u * y
    A[0::2, 8] = u

    A[1::2, 3] = -x
    A[1::2, 4] = -y
    A[1::2, 5] = -1
    A[1::2, 6] = v * x
    A[1::2, 7] = v * y
    A[1::2, 8] = v
    return A

def computeH(im1_pts: np.ndarray, im2_pts: np.ndarray) -> np.ndarray:
    """
    Return homography H (3x3)
    Uses plain DLT (no Hartley normalization).
    """
    A = build_A(im1_pts, im2_pts)   # (2N, 9)
    _, _, Vt = svd(A)
    h = Vt[-1, :]                   # smallest singular vector
    H = h.reshape(3, 3)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]             # scale for readability
    return H

# --------------------------- UI helpers --------------------------
def click_points(img, n: int, title: str) -> np.ndarray:
    """
    Show image, collect n clicks, press exit (close window).
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(img)
    ax.set_title(f"{title}")
    ax.axis("off")
    pts = plt.ginput(n, timeout=0)
    # draw markers before closing so user sees what they clicked
    pts = np.array(pts, dtype=float)
    ax.scatter(pts[:,0], pts[:,1], s=28, c="lime", marker="o", edgecolors="k")
    for i, (x, y) in enumerate(pts):
        ax.text(x+4, y-4, str(i+1), color="yellow", fontsize=9, weight="bold")
    plt.show()
    plt.close(fig)
    return pts

def show_matches(imgA, imgB, ptsA: np.ndarray, ptsB: np.ndarray):
    """
    Display side-by-side [A | B] with lines between correspondences.
    """
    # normalize to 3-channel for plotting consistency
    def to3(img):
        
        return np.stack([img, img, img], axis=2) if img.ndim == 2 else img

    A = to3(imgA)
    B = to3(imgB)

    h = max(A.shape[0], B.shape[0])
    # pad heights if needed
    if A.shape[0] < h:
        A = np.vstack([A, np.zeros((h - A.shape[0], A.shape[1], A.shape[2]), dtype=A.dtype)])
    if B.shape[0] < h:
        B = np.vstack([B, np.zeros((h - B.shape[0], B.shape[1], B.shape[2]), dtype=B.dtype)])

    canvas = np.hstack([A, B])
    shift_x = A.shape[1]

    ptsB_shift = ptsB.copy()
    ptsB_shift[:, 0] += shift_x

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(canvas)
    ax.axis("off")
    ax.set_title("Correspondences (left ↔ right)")

    ax.scatter(ptsA[:,0], ptsA[:,1], s=26, c="cyan", edgecolors="k")
    ax.scatter(ptsB_shift[:,0], ptsB_shift[:,1], s=26, c="orange", edgecolors="k")
    for i in range(len(ptsA)):
        ax.plot([ptsA[i,0], ptsB_shift[i,0]],
                [ptsA[i,1], ptsB_shift[i,1]],
                linewidth=1.2, alpha=0.9)
        ax.text(ptsA[i,0]+3, ptsA[i,1]-3, str(i+1), color="white", fontsize=9, weight="bold")
        ax.text(ptsB_shift[i,0]+3, ptsB_shift[i,1]-3, str(i+1), color="white", fontsize=9, weight="bold")
    plt.show()
    plt.close(fig)

def print_system_equations(src: np.ndarray, dst: np.ndarray):
    """
    Print the A h = 0 system (two rows per correspondence) and the numeric A.
    """
    A = build_A(src, dst)
    print("\n=== System of equations: A h = 0 ===")
    print(f"A shape: {A.shape} (2N x 9)")
    print("Unknowns h = [h11 h12 h13 h21 h22 h23 h31 h32 h33]^T\n")
    for i, ((x, y), (u, v)) in enumerate(zip(src, dst), start=1):
        r1 = f"[-{x:.4f}, -{y:.4f}, -1, 0, 0, 0, {u*x:.4f}, {u*y:.4f}, {u:.4f}] · h = 0"
        r2 = f"[0, 0, 0, -{x:.4f}, -{y:.4f}, -1, {v*x:.4f}, {v*y:.4f}, {v:.4f}] · h = 0"
        print(f"Pair {i}: (x,y)=({x:.2f},{y:.2f}) -> (u,v)=({u:.2f},{v:.2f})")
        print("  " + r1)
        print("  " + r2)
    print("\nA (rounded to 4 decimals):")
    with np.printoptions(precision=4, suppress=True):
        print(A)

# --------------------------------- main ---------------------------------
if __name__ == "__main__":

    A_path = "./media/room1.jpg"
    B_path = "./media/room2.jpg"

    num_points = 8

    # read images directly through plt.imread
    imgA = plt.imread(A_path)
    imgB = plt.imread(B_path)

    print("Click correspondences in Image A (left):")
    im1_pts = click_points(imgA, num_points, title="Image A")
    print("Click correspondences in Image B (right):")
    im2_pts = click_points(imgB, num_points, title="Image B")

    # show side-by-side with lines; no files saved
    show_matches(imgA, imgB, im1_pts, im2_pts)

    # print A h = 0, then compute/print H
    print_system_equations(im1_pts, im2_pts)
    H = computeH(im1_pts, im2_pts)

    print("\n=== Recovered homography H ===")
    with np.printoptions(precision=6, suppress=True):
        print(H)
