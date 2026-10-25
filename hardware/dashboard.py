"""Live matplotlib dashboard for hardware runs; the GUI must own the main thread (macOS)."""

import threading
import collections

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrow


class LiveDashboard:
    _WINDOW = 300

    def __init__(self, title: str = "Live Controller", track=None):
        self._title = title
        self._track  = track
        self._buf    = collections.deque(maxlen=self._WINDOW)
        self._lock   = threading.Lock()

    def push(self, step: int, cte: float, ir: tuple,
             dist_mm: int, vl: float, vr: float,
             x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        with self._lock:
            self._buf.append((step, cte, ir, dist_mm, vl, vr, x, y, theta))

    def start(self, controller_fn):
        """Run controller_fn in a daemon thread and block the main thread on the GUI."""
        t = threading.Thread(target=controller_fn, daemon=True)
        t.start()
        self._build_figure()
        self._anim = animation.FuncAnimation(    # unreferenced animations are garbage-collected
            self._fig, self._refresh,
            interval=150, blit=False, cache_frame_data=False,
        )
        plt.tight_layout()
        plt.show(block=True)

    def _build_figure(self):
        self._fig = plt.figure(figsize=(13, 9))
        self._fig.suptitle(self._title, fontsize=13, fontweight="bold")
        gs = gridspec.GridSpec(3, 3, figure=self._fig,
                               height_ratios=[1.6, 1.0, 1.0], hspace=0.45, wspace=0.35)

        ax_map = self._fig.add_subplot(gs[0, :])
        ax_map.set_title("Track vs estimated pose  (dead reckoning — drifts over time)")
        ax_map.set_aspect("equal")
        ax_map.grid(alpha=0.25)
        ax_map.set_xlabel("x (m)")
        ax_map.set_ylabel("y (m)")
        self._ax_map = ax_map

        if self._track is not None:
            sx = self._track._sx
            sy = self._track._sy
            ax_map.fill_between(sx, sy - 0.025, sy + 0.025,
                                color="#d4edda", alpha=0.5, label="line band (50 mm)")
            ax_map.plot(sx, sy, color="#28a745", linewidth=1.5, label="sim track center")
            pad = 0.3
            ax_map.set_xlim(sx.min() - pad, sx.max() + pad)
            ax_map.set_ylim(sy.min() - pad, sy.max() + pad)

        self._ln_trail, = ax_map.plot([], [], color="#4c9be8", linewidth=1.0,
                                       alpha=0.7, label="estimated path")
        self._pt_car,   = ax_map.plot([], [], "o", color="#e87c4c", markersize=8,
                                       label="car (est.)")
        self._car_arrow = None
        ax_map.legend(loc="upper right", fontsize=7)

        ax_cte = self._fig.add_subplot(gs[1, :2])
        ax_cte.set_title("Cross-Track Error (rolling)")
        ax_cte.set_ylabel("CTE (m)")
        ax_cte.set_ylim(-0.06, 0.06)
        ax_cte.axhline(0, color="green", linewidth=0.8, linestyle="--", alpha=0.6)
        ax_cte.grid(alpha=0.3)
        self._ax_cte  = ax_cte
        self._ln_cte, = ax_cte.plot([], [], "b-", linewidth=1.2)

        ax_vel = self._fig.add_subplot(gs[2, :2])
        ax_vel.set_title("Wheel Velocities")
        ax_vel.set_ylabel("v (m/s)")
        ax_vel.set_xlabel("Step")
        ax_vel.set_ylim(-0.6, 0.6)
        ax_vel.axhline(0, color="gray", linewidth=0.5)
        ax_vel.grid(alpha=0.3)
        self._ax_vel = ax_vel
        self._ln_vl, = ax_vel.plot([], [], color="#4c9be8", linewidth=1.0, label="Left")
        self._ln_vr, = ax_vel.plot([], [], color="#e87c4c", linewidth=1.0, label="Right")
        ax_vel.legend(loc="upper right", fontsize=8)

        ax_s = self._fig.add_subplot(gs[1:, 2])
        ax_s.set_xlim(0, 1)
        ax_s.set_ylim(0, 1)
        ax_s.axis("off")

        ax_s.text(0.5, 0.97, "IR Sensors", ha="center", va="top",
                  fontsize=9, color="gray")

        self._ir_patches = []
        for i, lbl in enumerate(["L", "C", "R"]):
            x = 0.18 + i * 0.32
            circle = plt.Circle((x, 0.84), 0.11, color="lightgray",
                                 transform=ax_s.transData)
            ax_s.add_patch(circle)
            ax_s.text(x, 0.84, lbl, ha="center", va="center",
                      fontsize=11, fontweight="bold")
            self._ir_patches.append(circle)

        self._txt_dist = ax_s.text(0.5, 0.65, "Dist:  — mm",    ha="center", fontsize=11)
        self._txt_cte  = ax_s.text(0.5, 0.55, "CTE:   — m",     ha="center", fontsize=11)
        self._txt_rms  = ax_s.text(0.5, 0.44, "RMS:   — m",     ha="center", fontsize=13,
                                    fontweight="bold", color="#222222")
        self._txt_step = ax_s.text(0.5, 0.30, "Step: 0",         ha="center", fontsize=9,
                                    color="gray")

        ax_spark = ax_s.inset_axes([0.05, 0.05, 0.90, 0.18])
        ax_spark.set_ylim(0, 0.05)
        ax_spark.set_title("RMS trend", fontsize=7, pad=2)
        ax_spark.tick_params(labelsize=6)
        ax_spark.grid(alpha=0.3)
        self._ax_spark = ax_spark
        self._ln_rms,  = ax_spark.plot([], [], "r-", linewidth=1.0)
        self._rms_hist: list[float] = []

    def _refresh(self, _frame):
        with self._lock:
            buf = list(self._buf)

        if not buf:
            return

        steps = np.array([d[0] for d in buf])
        ctes  = np.array([d[1] for d in buf])
        vls   = np.array([d[4] for d in buf])
        vrs   = np.array([d[5] for d in buf])
        xs    = np.array([d[6] for d in buf])
        ys    = np.array([d[7] for d in buf])
        last  = buf[-1]

        x0, x1 = steps[0], max(steps[-1], steps[0] + 1)

        self._ln_trail.set_data(xs, ys)
        self._pt_car.set_data([last[6]], [last[7]])
        if self._car_arrow is not None:
            self._car_arrow.remove()
        arrow_len = 0.15
        self._car_arrow = FancyArrow(
            last[6], last[7],
            arrow_len * np.cos(last[8]), arrow_len * np.sin(last[8]),
            width=0.04, color="#c0392b", zorder=5,
        )
        self._ax_map.add_patch(self._car_arrow)

        if self._track is not None:
            _, track_cte, _, _ = self._track.nearest(last[6], last[7])
            if not hasattr(self, "_txt_track_cte"):
                self._txt_track_cte = self._ax_map.text(
                    0.02, 0.98, "", transform=self._ax_map.transAxes,
                    va="top", fontsize=8, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )
            self._txt_track_cte.set_text(
                f"IR CTE: {last[1]:+.4f} m\n"
                f"Map CTE: {track_cte:+.4f} m\n"
                f"pose: ({last[6]:+.2f}, {last[7]:+.2f})"
            )

        self._ln_cte.set_data(steps, ctes)
        self._ax_cte.set_xlim(x0, x1)

        self._ln_vl.set_data(steps, vls)
        self._ln_vr.set_data(steps, vrs)
        self._ax_vel.set_xlim(x0, x1)

        for patch, active in zip(self._ir_patches, last[2]):
            patch.set_facecolor("#f4a26f" if active else "lightgray")

        dist    = last[3]
        rms     = float(np.sqrt(np.mean(ctes ** 2)))
        dist_s  = f"{dist} mm" if dist < 9000 else "—"
        self._txt_dist.set_text(f"Dist:  {dist_s}")
        self._txt_cte.set_text(f"CTE:  {last[1]:+.4f} m")
        self._txt_rms.set_text(f"RMS: {rms:.4f} m")
        self._txt_step.set_text(f"Step: {last[0]}")

        self._rms_hist.append(rms)
        if len(self._rms_hist) > 200:
            self._rms_hist.pop(0)
        self._ln_rms.set_data(range(len(self._rms_hist)), self._rms_hist)
        self._ax_spark.set_xlim(0, max(len(self._rms_hist), 1))
        self._ax_spark.set_ylim(0, max(max(self._rms_hist) * 1.2, 0.005))
