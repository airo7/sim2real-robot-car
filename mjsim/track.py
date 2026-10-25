import numpy as np
from scipy.interpolate import splprep, splev


class Track:
    N_SAMPLES = 1000

    def __init__(self, waypoints: np.ndarray):
        """waypoints: (N, 2) control points of a closed loop, in metres."""
        self.waypoints = waypoints
        self._build_spline()

    def _build_spline(self):
        pts = self.waypoints
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=True)
        self.tck = tck

        t = np.linspace(0, 1, self.N_SAMPLES, endpoint=False)
        xy = splev(t, tck)
        dxy = splev(t, tck, der=1)

        self._sx = np.array(xy[0])
        self._sy = np.array(xy[1])
        self._tangent = np.arctan2(dxy[1], dxy[0])

        dx = np.diff(np.append(self._sx, self._sx[0]))
        dy = np.diff(np.append(self._sy, self._sy[0]))
        seg_len = np.sqrt(dx**2 + dy**2)
        self.total_length = seg_len.sum()
        self.sample_res = self.total_length / self.N_SAMPLES

    def nearest(self, x: float, y: float):
        """Return (idx, cte, theta_ref, progress_frac); cte positive = left of travel direction."""
        dx = self._sx - x
        dy = self._sy - y
        idx = int(np.argmin(dx**2 + dy**2))

        theta_ref = self._tangent[idx]
        ex = x - self._sx[idx]
        ey = y - self._sy[idx]
        cte = -ex * np.sin(theta_ref) + ey * np.cos(theta_ref)

        return idx, cte, theta_ref, idx / self.N_SAMPLES

    def reference_ahead(self, idx: int, n: int, step: int = 3):
        """Return an (n, 3) array of (x, y, theta) poses, every `step` samples ahead of idx."""
        indices = [(idx + i * step) % self.N_SAMPLES for i in range(1, n + 1)]
        return np.stack([
            self._sx[indices],
            self._sy[indices],
            self._tangent[indices],
        ], axis=1)

    @property
    def sampled(self) -> np.ndarray:
        return np.stack([self._sx, self._sy], axis=1)

    @classmethod
    def make_oval(cls, cx=0., cy=0., rx=2.0, ry=1.2, n=60):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return cls(np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], axis=1))

    @classmethod
    def make_figure8(cls, cx=0., cy=0., scale=1.5, n=120):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = cx + scale * np.sin(t)
        y = cy + scale * np.sin(t) * np.cos(t)
        return cls(np.stack([x, y], axis=1))

    @classmethod
    def make_rect(cls, cx=0., cy=0., width=1.2, height=0.8,
                  corner_radius=0.2, n_per_edge=22, n_per_corner=12):
        """CCW rounded rectangle; sharp 90° corners can tip the real car over."""
        hw, hh = width / 2.0, height / 2.0
        r = min(corner_radius, hw, hh)
        pts = []

        def edge(x0, y0, x1, y1):
            for t in np.linspace(0, 1, n_per_edge, endpoint=False):
                pts.append([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t])

        def arc(acx, acy, a0, a1):
            for a in np.linspace(a0, a1, n_per_corner, endpoint=False):
                pts.append([acx + r * np.cos(a), acy + r * np.sin(a)])

        edge(cx - hw + r, cy - hh, cx + hw - r, cy - hh)
        arc(cx + hw - r, cy - hh + r, -np.pi / 2, 0.0)
        edge(cx + hw, cy - hh + r, cx + hw, cy + hh - r)
        arc(cx + hw - r, cy + hh - r, 0.0, np.pi / 2)
        edge(cx + hw - r, cy + hh, cx - hw + r, cy + hh)
        arc(cx - hw + r, cy + hh - r, np.pi / 2, np.pi)
        edge(cx - hw, cy + hh - r, cx - hw, cy - hh + r)
        arc(cx - hw + r, cy - hh + r, np.pi, 3 * np.pi / 2)

        return cls(np.array(pts, dtype=np.float64))

    @classmethod
    def from_name(cls, name: str):
        factories = {
            "rect":    cls.make_rect,
            "oval":    cls.make_oval,
            "figure8": cls.make_figure8,
        }
        if name not in factories:
            raise ValueError(f"Unknown track {name!r}; choose from {list(factories)}")
        return factories[name]()
