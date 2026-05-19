"""
One-time calibration: align the experimental plate photo with the
simulation coordinate system by clicking the antibiotic disk centres.

Run once:

    python calibrate.py

It opens the .tiff, you click each disk centre IN THE ORDER they are
listed in main_orso.DISKS, it fits a similarity transform
(scale + rotation + translation, with the image y-flip handled), prints
the fit error, shows a check plot, and writes calibration.json.

main_orso.py then loads calibration.json to overlay the simulation
isolines onto the photo.
"""

import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from main_orso import DISKS, IMAGE_PATH, CALIBRATION_PATH


def fit_similarity(mm_xy, px_xy, image_height):
    """
    Fit pixel = A * (x_mm + i*y_mm) + B  in a y-up "math" frame.

    Complex linear least squares over all disks. A encodes scale and
    rotation, B the translation. Over-determined for >=3 disks, so the
    residual is a real measure of alignment quality.

    Returns (A, B, rms_pixels, predicted_px_xy).
    """
    z = mm_xy[:, 0] + 1j * mm_xy[:, 1]                       # mm
    w = px_xy[:, 0] + 1j * (image_height - px_xy[:, 1])      # px, y-up

    # [z, 1] @ [A, B]^T = w
    M = np.column_stack([z, np.ones_like(z)])
    (A, B), *_ = np.linalg.lstsq(M, w, rcond=None)

    w_pred = A * z + B
    pred_px = np.column_stack([
        np.real(w_pred),
        image_height - np.imag(w_pred),
    ])
    rms = np.sqrt(np.mean(np.sum((pred_px - px_xy) ** 2, axis=1)))
    return A, B, rms, pred_px


def main():
    img = mpimg.imread(IMAGE_PATH)
    H = img.shape[0]

    mm_xy = np.array([[d["x_mm"], d["y_mm"]] for d in DISKS], dtype=float)
    n = len(DISKS)

    # --- Click the disk centres -------------------------------------
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img, cmap="gray")
    order = "  ->  ".join(
        f"({d['x_mm']}, {d['y_mm']}) mm" for d in DISKS
    )
    ax.set_title(
        f"Click the {n} disk centres IN THIS ORDER:\n{order}",
        fontsize=10,
    )
    print(f"Click the {n} disk centres in this order:")
    for i, d in enumerate(DISKS, 1):
        print(f"  {i}. ({d['x_mm']}, {d['y_mm']}) mm  (mass {d['mass']})")

    clicks = plt.ginput(n, timeout=0)
    plt.close(fig)

    if len(clicks) != n:
        raise SystemExit(
            f"Expected {n} clicks, got {len(clicks)}. Re-run calibrate.py."
        )
    px_xy = np.array(clicks, dtype=float)

    # --- Fit ---------------------------------------------------------
    A, B, rms, pred_px = fit_similarity(mm_xy, px_xy, H)
    scale_px_per_mm = abs(A)
    print(f"\nFit: scale = {scale_px_per_mm:.3f} px/mm, "
          f"rotation = {np.degrees(np.angle(A)):.2f} deg")
    print(f"RMS alignment error = {rms:.2f} px "
          f"(~{rms / scale_px_per_mm:.3f} mm)")
    if rms / scale_px_per_mm > 1.0:
        print("WARNING: error > 1 mm. Re-run and click disk centres "
              "more precisely, or check the click order.")

    # --- Check plot: clicked vs predicted + mapped dish boundary ----
    def mm_to_px(x_mm, y_mm):
        w = A * (np.asarray(x_mm) + 1j * np.asarray(y_mm)) + B
        return np.real(w), H - np.imag(w)

    theta = np.linspace(0, 2 * np.pi, 200)
    ring_px = mm_to_px(45 + 45 * np.cos(theta), 45 + 45 * np.sin(theta))

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img, cmap="gray")
    ax.plot(ring_px[0], ring_px[1], "c-", lw=1,
            label="mapped 90 mm dish")
    ax.plot(px_xy[:, 0], px_xy[:, 1], "yx", ms=12, mew=2,
            label="your clicks")
    ax.plot(pred_px[:, 0], pred_px[:, 1], "r+", ms=14, mew=2,
            label="fitted disk centres")
    ax.legend(loc="upper right")
    ax.set_title(f"Calibration check  (RMS {rms:.2f} px). "
                 "Close window to save.")
    plt.show()

    # --- Save --------------------------------------------------------
    calib = {
        "image": IMAGE_PATH,
        "A_re": float(np.real(A)),
        "A_im": float(np.imag(A)),
        "B_re": float(np.real(B)),
        "B_im": float(np.imag(B)),
        "H": int(H),
        "rms_px": float(rms),
        "scale_px_per_mm": float(scale_px_per_mm),
        "disks_mm": mm_xy.tolist(),
        "clicks_px": px_xy.tolist(),
    }
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\nSaved calibration -> {CALIBRATION_PATH}")
    print("Now run `python main_orso.py` to see the overlay.")


if __name__ == "__main__":
    main()
