import os

import numpy as np
import matplotlib.pyplot as plt

import Diffusion_constant_calc  # prints its theoretical D on import


# ============================================================
#  CONFIG  --  edit here, then run `python main_orso.py`
#  (calibrate.py imports DISKS / IMAGE_PATH / CALIBRATION_PATH)
# ============================================================

_HERE = os.path.dirname(__file__)

# --- Geometry & grid -----------------------------------------
DISH_SIZE_MM = 90
AGAR_DEPTH_MM = 6.0
DISH_RADIUS_MM = None   # None -> inscribed circle (DISH_SIZE_MM/2)

DX_MM = 0.5             # grid step; 0.25 = finer & slower
CFL_SAFETY = 0.8        # fraction of the stability limit (<1)

# --- Physics -------------------------------------------------
D_FROM_THEORY = Diffusion_constant_calc.D_ampicillin_mm2_hour  # ~0.554
DIFFUSION_COEFFICIENT = D_FROM_THEORY   # or a number, e.g. 5.0

# --- Time ----------------------------------------------------
TOTAL_TIME_HOURS = 17.5
SAVE_EVERY_HOURS = 1.0

# --- Antibiotic disks ----------------------------------------
# ORDER defines the click order in calibrate.py.
DISKS = [
    dict(x_mm=50.9, y_mm=66.6, radius_mm=3.9, mass=2),
    dict(x_mm=23.8, y_mm=40.6, radius_mm=3.9, mass=10),
    dict(x_mm=61.8, y_mm=30.7, radius_mm=3.9, mass=50),
]

# Measured inhibition-zone radii [mm], one per disk above (read off
# the plate photo: disk centre -> edge of the clear halo). Entry None
# falls back to that disk's own radius as a placeholder.
MEASURED_ZONE_RADII_MM = [None, None, None]

# --- Image overlay -------------------------------------------
IMAGE_PATH = os.path.join(_HERE, "Disk Diffusion Plate.tiff")
CALIBRATION_PATH = os.path.join(_HERE, "calibration.json")
ISOLINE_LEVELS = [0, 1e-6, 2e-3, 1e-2, 0.1, 1]   # None -> auto-placed
HEATMAP_ALPHA = 0.2     # None -> isolines only
ISOLINE_COLOR = "white"  # contour line/label colour (any matplotlib colour)
ISOLINE_WIDTH = 0.8      # contour line thickness
SHOW_AXES = False       # False -> hide pixel x/y axes for a clean figure
CONCENTRATION_UNIT = "ug/mm^2"   # colour-bar unit label (mass / area)

# --- Optional progression video ------------------------------
MAKE_VIDEO = 1         # True -> animate saved frames on the photo
VIDEO_FPS = 4              # frames per second (one frame per saved dt)
VIDEO_SAVE_PATH = ["video.mp4", "video.gif"]  # str or list; None = view only
VIDEO_DPI = 200            # output resolution (dots per inch)
VIDEO_FIGSIZE_IN = 9       # figure size in inches (square) -> px = this * DPI

# ============================================================


def _fmt_level(v):
    """Contour-label formatter: 0.1 -> '0.1', 1e-6 -> '0.000001',
    1 -> '1' (plain decimals, no trailing zeros, no sci-notation)."""
    if v == 0:
        return "0"
    s = f"{v:.12f}".rstrip("0").rstrip(".")
    return s if s else "0"


class PetriDishDiffusionSolver:
    def __init__(
        self,
        dish_size_mm=90,
        dx_mm=1.0,
        diffusion_coefficient=1.0,  # mm^2 / hour
        agar_depth_mm=6.0,          # agar layer thickness (mm)
        dish_radius_mm=None,        # default: inscribed circle
        dt_hours=None,
        cfl_safety=0.8              # fraction of the stability limit
    ):
        self.dish_size_mm = dish_size_mm
        self.dx = dx_mm
        self.dy = dx_mm
        self.D = diffusion_coefficient
        self.agar_depth_mm = agar_depth_mm

        self.nx = int(dish_size_mm / dx_mm)
        self.ny = int(dish_size_mm / dx_mm)

        # 2D explicit-diffusion stability limit: dt <= dx^2 / (4 D).
        # Run at cfl_safety * limit so the checkerboard mode is damped
        # (at exactly the limit it is only marginally stable).
        max_dt = self.dx**2 / (4 * self.D)
        if dt_hours is None:
            self.dt = cfl_safety * max_dt
        else:
            self.dt = dt_hours
        if self.dt > max_dt:
            raise ValueError(
                f"dt is too large and unstable. Use dt <= {max_dt:.4f} hours."
            )

        # Circular petri dish: the no-flux wall is the DISH, not the
        # square grid. In the mm frame the dish is the inscribed circle
        # of radius dish_size_mm/2 centred at the domain centre -- the
        # exact same circle calibrate.py / plot_overlay map onto the
        # photo, so the simulation wall and the overlay ring coincide.
        self.dish_radius_mm = (
            dish_size_mm / 2 if dish_radius_mm is None else dish_radius_mm
        )
        cx = cy = dish_size_mm / 2
        xs = np.arange(self.nx) * self.dx
        ys = np.arange(self.ny) * self.dy
        X, Y = np.meshgrid(xs, ys)
        self.inside = (
            (X - cx) ** 2 + (Y - cy) ** 2 <= self.dish_radius_mm ** 2
        )
        # Reflective-neighbour masks for the finite-difference no-flux
        # wall: True where the neighbour in that direction is also in the
        # dish. Where it is not, step() substitutes the centre value so
        # the gradient (hence the flux) across the curved wall is zero.
        in_pad = np.pad(self.inside, 1, constant_values=False)
        self._up_in = in_pad[:-2, 1:-1]
        self._down_in = in_pad[2:, 1:-1]
        self._left_in = in_pad[1:-1, :-2]
        self._right_in = in_pad[1:-1, 2:]

        self.C = np.zeros((self.ny, self.nx), dtype=float)

    def add_disk(self, x_mm, y_mm, radius_mm, mass):
        """
        Add an antibiotic disk: `mass` is the total loaded amount, spread
        uniformly through the disk-shaped agar volume beneath it.

        concentration = mass / (pi * radius_mm^2 * agar_depth_mm)

        This is a true volumetric concentration (e.g. ug -> ug/mm^3 =
        mg/mL) and is INDEPENDENT of the grid resolution dx, so the
        value you read at a zone edge to infer the MIC is physical and
        reproducible.
        """
        volume_mm3 = np.pi * radius_mm**2 # * self.agar_depth_mm but we use area concentration for a 2D model, so ignore depth here
        concentration = mass / volume_mm3

        xs = np.arange(self.nx) * self.dx
        ys = np.arange(self.ny) * self.dy
        X, Y = np.meshgrid(xs, ys)
        mask = (
            ((X - x_mm)**2 + (Y - y_mm)**2 <= radius_mm**2) & self.inside
        )

        self.C[mask] = concentration

    def step(self):
        """
        One Forward-Euler diffusion step, finite-difference form
        (standard 5-point Laplacian).

        No-flux wall on the circular dish: any neighbour that lies
        outside the dish is replaced by the centre value, giving a zero
        one-sided gradient (hence zero flux) across the curved wall.
        Cells outside the dish are held at zero. The inside/outside
        stencil terms cancel pairwise, so this is also mass conserving
        to machine precision -- it is equivalent to the finite-VOLUME
        variant for this constant-D problem, just written as a pointwise
        Laplacian instead of a face-flux divergence.
        """
        C = self.C

        # Edge-padding gives a zero-gradient at the square grid border
        # (those cells are outside the inscribed dish anyway).
        Cp = np.pad(C, 1, mode="edge")
        up = Cp[:-2, 1:-1]
        down = Cp[2:, 1:-1]
        left = Cp[1:-1, :-2]
        right = Cp[1:-1, 2:]

        # Reflective no-flux at the circular dish wall.
        up = np.where(self._up_in, up, C)
        down = np.where(self._down_in, down, C)
        left = np.where(self._left_in, left, C)
        right = np.where(self._right_in, right, C)

        laplacian = (
            (up - 2 * C + down) / self.dy**2
            +
            (left - 2 * C + right) / self.dx**2
        )

        C_new = C + self.D * self.dt * laplacian
        C_new[~self.inside] = 0.0
        self.C = C_new

    def run(self, total_time_hours, save_every_hours=1.0):
        """
        Returns:
        times: array of saved time points
        concentrations: array with shape [time, y, x]
        """
        import time as _time

        total_steps = int(total_time_hours / self.dt)
        save_interval = max(1, int(save_every_hours / self.dt))
        # Wall-clock interval between progress prints (seconds).
        progress_every_s = 5.0

        saved_times = []
        saved_concentrations = []

        wall_start = _time.perf_counter()
        last_print = wall_start
        print(
            f"Running diffusion: {total_steps} steps, dt={self.dt:.5f} h, "
            f"grid {self.ny}x{self.nx}"
        )

        for step in range(total_steps + 1):
            sim_time = step * self.dt

            if step % save_interval == 0:
                saved_times.append(sim_time)
                saved_concentrations.append(self.C.copy())

            now = _time.perf_counter()
            if now - last_print >= progress_every_s or step == total_steps:
                last_print = now
                pct = 100.0 * step / total_steps
                elapsed = now - wall_start
                eta = elapsed / step * (total_steps - step) if step else 0.0
                print(
                    f"\r  {pct:5.1f}%  |  sim t = {sim_time:5.2f}/"
                    f"{total_time_hours:g} h  |  elapsed {elapsed:5.1f}s  "
                    f"ETA {eta:5.1f}s",
                    end="",
                    flush=True,
                )

            self.step()

        print(
            f"\r  100.0%  |  done in "
            f"{_time.perf_counter() - wall_start:.1f}s"
            + " " * 20
        )

        return np.array(saved_times), np.array(saved_concentrations)

    def plot(self, concentration, title="Ampicillin concentration"):
        plt.imshow(
            concentration,
            origin="lower",
            extent=[0, self.dish_size_mm, 0, self.dish_size_mm]
        )
        plt.colorbar(label="Concentration")
        plt.xlabel("x position [mm]")
        plt.ylabel("y position [mm]")
        plt.title(title)
        plt.show()

    def plot_isolines(
        self,
        concentration,
        levels=None,
        title="Ampicillin concentration isolines",
        show_filled_background=True
    ):
        """
        Plots isolines for specified concentration levels.

        Parameters
        ----------
        concentration : 2D numpy array
            Concentration field at one time point.
        levels : list of float, or None
            Concentration values for which isolines are drawn. If None
            (default), matplotlib auto-places the contour levels.
        title : str
            Plot title.
        show_filled_background : bool
            If True, also shows a faint concentration heatmap underneath.
        """

        x = np.linspace(0, self.dish_size_mm, self.nx)
        y = np.linspace(0, self.dish_size_mm, self.ny)
        X, Y = np.meshgrid(x, y)

        plt.figure()

        if show_filled_background:
            plt.imshow(
                concentration,
                origin="lower",
                extent=[0, self.dish_size_mm, 0, self.dish_size_mm],
                alpha=0.35
            )
            plt.colorbar(label="Concentration")

        if levels is None:
            # Auto-placed contour levels.
            contours = plt.contour(X, Y, concentration)
        else:
            contours = plt.contour(X, Y, concentration, levels=levels)

        plt.clabel(contours, inline=True, fontsize=8, fmt=_fmt_level)

        plt.xlabel("x position [mm]")
        plt.ylabel("y position [mm]")
        plt.title(title)
        plt.axis("equal")
        plt.show()

    def concentration_at(self, concentration, x_mm, y_mm):
        """
        Returns the simulated concentration at a physical (x, y) location in mm.

        Use this with a zone-of-inhibition radius measured from the experiment
        image: sample the concentration at the zone edge to infer the effective
        inhibitory concentration (MIC) of ampicillin.
        """
        x0 = int(round(x_mm / self.dx))
        y0 = int(round(y_mm / self.dy))
        x0 = min(max(x0, 0), self.nx - 1)
        y0 = min(max(y0, 0), self.ny - 1)
        return concentration[y0, x0]

    def concentration_at_radius(self, concentration, x_mm, y_mm, radius_mm,
                                n_samples=360):
        """
        Mean (and std) simulated concentration on a circle of given radius
        around a disk centre. Average a measured zone-of-inhibition radius
        over all angles to get a robust MIC estimate.
        """
        angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        xs = x_mm + radius_mm * np.cos(angles)
        ys = y_mm + radius_mm * np.sin(angles)
        vals = np.array([
            self.concentration_at(concentration, x, y)
            for x, y in zip(xs, ys)
        ])
        return vals.mean(), vals.std()

    @staticmethod
    def _mm_to_px(calib, x_mm, y_mm):
        """
        Map simulation coordinates (mm) to image pixels using the
        similarity transform stored in `calib` (see calibrate.py).

        The fit is done in a y-up "math" frame; we flip back to the
        image's y-down pixel convention with H (image height in px).
        """
        A = complex(calib["A_re"], calib["A_im"])
        B = complex(calib["B_re"], calib["B_im"])
        w = A * (np.asarray(x_mm) + 1j * np.asarray(y_mm)) + B
        px = np.real(w)
        py = calib["H"] - np.imag(w)
        return px, py

    def plot_overlay(
        self,
        concentration,
        image_path,
        calib,
        levels=None,
        disks_mm=None,
        heatmap_alpha=None,
        cmap="viridis",
        isoline_color="white",
        isoline_width=1.0,
        show_axes=True,
        conc_unit=None,
        title="Simulation isolines over experimental plate"
    ):
        """
        Overlay the concentration isolines on the experimental .tiff,
        aligned via the calibration produced by calibrate.py.

        heatmap_alpha : float or None
            If set (0-1), also draw the concentration field as a
            translucent colour map on top of the photo at this opacity.
            None (default) = isolines only.
        cmap : str
            Colormap for the optional heatmap.
        show_axes : bool
            False hides the pixel x/y axes for a clean figure.
        conc_unit : str or None
            Unit appended to the colour-bar label, e.g. "ug/mm^2".
        """
        import matplotlib.image as mpimg

        img = mpimg.imread(image_path)

        x = np.linspace(0, self.dish_size_mm, self.nx)
        y = np.linspace(0, self.dish_size_mm, self.ny)
        Xmm, Ymm = np.meshgrid(x, y)
        PX, PY = self._mm_to_px(calib, Xmm, Ymm)

        _, ax = plt.subplots()
        ax.imshow(img, cmap="gray")

        # Optional translucent concentration heatmap. pcolormesh on the
        # transformed grid so it follows the calibration rotation/scale.
        if heatmap_alpha is not None:
            mesh = ax.pcolormesh(
                PX, PY, concentration,
                cmap=cmap, alpha=heatmap_alpha,
                shading="auto", zorder=2
            )
            label = (
                f"Concentration [{conc_unit}]" if conc_unit
                else "Concentration"
            )
            plt.colorbar(mesh, ax=ax, label=label)

        if levels is None:
            cs = ax.contour(
                PX, PY, concentration,
                colors=isoline_color, linewidths=isoline_width
            )
        else:
            cs = ax.contour(
                PX, PY, concentration, levels=levels,
                colors=isoline_color, linewidths=isoline_width
            )
        ax.clabel(cs, inline=True, fontsize=8, fmt=_fmt_level)

        # Dish boundary and antibiotic disks, drawn as grey dashed
        # circles. If the calibration is good these should line up with
        # the real dish rim and disks in the photo.
        theta = np.linspace(0, 2 * np.pi, 200)

        def dashed_circle(cx_mm, cy_mm, r_mm):
            cx = cx_mm + r_mm * np.cos(theta)
            cy = cy_mm + r_mm * np.sin(theta)
            px, py = self._mm_to_px(calib, cx, cy)
            ax.plot(px, py, color="grey", lw=1.2, ls="--", zorder=3)

        # Modelled petri dish: inscribed circle of the square domain.
        dashed_circle(
            self.dish_size_mm / 2, self.dish_size_mm / 2,
            self.dish_size_mm / 2
        )

        if disks_mm:
            for dx_mm, dy_mm, r_mm in disks_mm:
                dashed_circle(dx_mm, dy_mm, r_mm)

        ax.set_title(title)
        if show_axes:
            ax.set_xlabel("image x [px]")
            ax.set_ylabel("image y [px]")
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        plt.show()

    def animate_overlay(
        self,
        times,
        concentration_series,
        image_path,
        calib,
        levels=None,
        disks_mm=None,
        heatmap_alpha=None,
        cmap="viridis",
        isoline_color="white",
        isoline_width=1.0,
        show_axes=True,
        conc_unit=None,
        fps=4,
        dpi=200,
        figsize_in=9,
        save_path=None,
        title="Diffusion progression over experimental plate"
    ):
        """
        Optional video viewer: play every saved frame of
        `concentration_series` (shape [t, y, x], as returned by run())
        on top of the experimental photo, so you can watch the front
        spread. Frame spacing = SAVE_EVERY_HOURS (lower it for a
        smoother movie).

        show_axes : bool
            False hides the pixel x/y axes for a clean figure.
        conc_unit : str or None
            Accepted for signature parity with plot_overlay; the
            animation has no colour bar so it is unused here.
        dpi : int
            Output resolution. Pixel size = figsize_in * dpi per side.
        figsize_in : float
            Square figure size in inches.
        save_path : None | str | list[str]
            One or several output paths; the writer is chosen per file
            extension. ".gif" uses Pillow; ".mp4" uses ffmpeg (the
            bundled imageio-ffmpeg binary is used automatically if no
            system ffmpeg is on PATH). None just opens the viewer.
        """
        _ = conc_unit  # no colour bar in the animation
        import matplotlib.image as mpimg
        from matplotlib.animation import FuncAnimation, PillowWriter

        img = mpimg.imread(image_path)
        series = np.asarray(concentration_series)

        x = np.linspace(0, self.dish_size_mm, self.nx)
        y = np.linspace(0, self.dish_size_mm, self.ny)
        Xmm, Ymm = np.meshgrid(x, y)
        PX, PY = self._mm_to_px(calib, Xmm, Ymm)

        # Fixed colour scale across all frames (99th pct of positive
        # values) so the spreading front stays visible instead of the
        # t=0 disk peak saturating everything.
        pos = series[series > 0]
        vmax = float(np.percentile(pos, 99)) if pos.size else 1.0

        theta = np.linspace(0, 2 * np.pi, 200)

        def dashed_circle(ax, cx_mm, cy_mm, r_mm):
            cx = cx_mm + r_mm * np.cos(theta)
            cy = cy_mm + r_mm * np.sin(theta)
            px, py = self._mm_to_px(calib, cx, cy)
            ax.plot(px, py, color="grey", lw=1.2, ls="--", zorder=3)

        fig, ax = plt.subplots(figsize=(figsize_in, figsize_in), dpi=dpi)

        def draw(k):
            ax.clear()
            ax.imshow(img, cmap="gray")
            C = series[k]
            if heatmap_alpha is not None:
                ax.pcolormesh(
                    PX, PY, C, cmap=cmap, alpha=heatmap_alpha,
                    shading="auto", vmin=0.0, vmax=vmax, zorder=2
                )
            if levels is None:
                cs = ax.contour(
                    PX, PY, C,
                    colors=isoline_color, linewidths=isoline_width
                )
            else:
                cs = ax.contour(
                    PX, PY, C, levels=levels,
                    colors=isoline_color, linewidths=isoline_width
                )
            ax.clabel(cs, inline=True, fontsize=8, fmt=_fmt_level)

            dashed_circle(
                ax, self.dish_size_mm / 2, self.dish_size_mm / 2,
                self.dish_size_mm / 2
            )
            if disks_mm:
                for dx_mm, dy_mm, r_mm in disks_mm:
                    dashed_circle(ax, dx_mm, dy_mm, r_mm)

            ax.set_title(f"{title}  --  t = {times[k]:.2f} h")
            if show_axes:
                ax.set_xlabel("image x [px]")
                ax.set_ylabel("image y [px]")
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

        anim = FuncAnimation(
            fig, draw, frames=len(series),
            interval=1000.0 / fps, blit=False
        )

        if save_path:
            paths = (
                [save_path] if isinstance(save_path, str)
                else list(save_path)
            )

            # Use the bundled imageio-ffmpeg binary if no system ffmpeg.
            if any(str(p).lower().endswith(".mp4") for p in paths):
                try:
                    import matplotlib as _mpl
                    import imageio_ffmpeg
                    _mpl.rcParams["animation.ffmpeg_path"] = (
                        imageio_ffmpeg.get_ffmpeg_exe()
                    )
                except Exception:
                    pass  # fall back to system ffmpeg if present

            for p in paths:
                ext = str(p).lower().rsplit(".", 1)[-1]
                try:
                    if ext == "gif":
                        anim.save(
                            p, writer=PillowWriter(fps=fps), dpi=dpi
                        )
                    else:  # mp4 / mov / ... -> ffmpeg
                        anim.save(p, writer="ffmpeg", fps=fps, dpi=dpi)
                    print(f"Saved animation -> {p}")
                except Exception as e:
                    print(f"Could not save {p}: {e}")

        plt.show()
        return anim  # keep a reference so it is not garbage-collected


if __name__ == "__main__":
    import json

    solver = PetriDishDiffusionSolver(
        dish_size_mm=DISH_SIZE_MM,
        dx_mm=DX_MM,
        diffusion_coefficient=DIFFUSION_COEFFICIENT,
        agar_depth_mm=AGAR_DEPTH_MM,
        dish_radius_mm=DISH_RADIUS_MM,
        cfl_safety=CFL_SAFETY,
    )

    for d in DISKS:
        solver.add_disk(**d)

    times, concentrations = solver.run(
        total_time_hours=TOTAL_TIME_HOURS,
        save_every_hours=SAVE_EVERY_HOURS,
    )

    # concentration at every point and every saved time:
    # concentrations[time_index, y, x]

    print("Times saved:", times)
    print("Output shape:", concentrations.shape)

    # Mass conservation check: total mass = sum(C) * cell volume, and
    # it should equal the loaded mass (no-flux circular wall, no decay).
    cell_volume = solver.dx * solver.dy # * solver.agar_depth_mm
    loaded = sum(d["mass"] for d in DISKS)
    m0 = concentrations[0].sum() * cell_volume
    m1 = concentrations[-1].sum() * cell_volume
    print(f"Loaded mass        : {loaded:.6g}")
    print(f"Total mass t=0     : {m0:.6g}")
    print(f"Total mass t={times[-1]:.2f} h : {m1:.6g}  "
          f"(drift {(m1 - m0) / m0:.2e})")

    # Initial (t=0) loaded concentration at each disk centre.
    initial = concentrations[0]
    for d in DISKS:
        c0 = solver.concentration_at(initial, d["x_mm"], d["y_mm"])
        print(
            f"Disk mass={d['mass']}: C at centre, t=0 = {c0:.4g} "
            f"(= mass / (pi r^2 depth))"
        )

    # --- Infer effective inhibitory concentration from measured zones ---
    # Uses MEASURED_ZONE_RADII_MM from CONFIG (None -> disk radius).
    final = concentrations[-1]
    for d, r_meas in zip(DISKS, MEASURED_ZONE_RADII_MM):
        radius = d["radius_mm"] if r_meas is None else r_meas
        c_mean, c_std = solver.concentration_at_radius(
            final, d["x_mm"], d["y_mm"], radius
        )
        tag = " (placeholder)" if r_meas is None else ""
        print(
            f"Disk mass={d['mass']}: C at r={radius} mm{tag} "
            f"after {times[-1]:.2f} h = {c_mean:.4g} +/- {c_std:.2g}"
        )

#    # Plot final concentration after 17.5 hours: heatmap + isolines
#    solver.plot(
#        concentrations[-1],
#        title="Ampicillin concentration after 17.5 h"
#    )
#    # Pass levels=[...] for manual isolines; omit for auto-placed levels.
#    solver.plot_isolines(
#        concentrations[-1],
#        levels=None,
#        title="Ampicillin isolines after 17.5 h"
#    )

    # Overlay on the experimental photo (requires calibration.json:
    # run `python calibrate.py` once first).
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH) as f:
            calib = json.load(f)
        disks_xyr = [
            (d["x_mm"], d["y_mm"], d["radius_mm"]) for d in DISKS
        ]
        overlay_title = (
            "Antibiotic diffusion Simulation overlayed experimental "
            f"result after {TOTAL_TIME_HOURS:g}h"
        )
        if MAKE_VIDEO:
            _anim = solver.animate_overlay(
                times,
                concentrations,
                IMAGE_PATH,
                calib,
                levels=ISOLINE_LEVELS,
                disks_mm=disks_xyr,
                heatmap_alpha=HEATMAP_ALPHA,
                isoline_color=ISOLINE_COLOR,
                isoline_width=ISOLINE_WIDTH,
                show_axes=SHOW_AXES,
                conc_unit=CONCENTRATION_UNIT,
                fps=VIDEO_FPS,
                dpi=VIDEO_DPI,
                figsize_in=VIDEO_FIGSIZE_IN,
                save_path=VIDEO_SAVE_PATH,
            )
        else:
            solver.plot_overlay(
                concentrations[-1],
                IMAGE_PATH,
                calib,
                levels=ISOLINE_LEVELS,
                disks_mm=disks_xyr,
                heatmap_alpha=HEATMAP_ALPHA,
                isoline_color=ISOLINE_COLOR,
                isoline_width=ISOLINE_WIDTH,
                show_axes=SHOW_AXES,
                conc_unit=CONCENTRATION_UNIT,
                title=overlay_title,
            )
    else:
        print(
            f"\nNo calibration found at {CALIBRATION_PATH}. "
            "Run `python calibrate.py` to enable the image overlay."
        )

