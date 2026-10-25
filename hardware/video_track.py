"""
Measure the car's pose relative to the tape from a phone video of a real run.

Works with a hand-held camera that follows the car: everything is measured in the
car's own frame (tape offset at the IR sensors, tape angle to the car axis), so
camera motion cancels out. Scale comes from the 231 mm plate length.

Writes <video>.track.csv and an annotated <video>.annotated.mp4. With --log, also
finds the video↔log time offset by cross-correlating the tape side seen in the video
with the IR centroid in the run log.

Usage:
    python3 -m hardware.video_track VIDEO [--log hardware/logs/run_….csv]
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

PLATE_MM = 231.0
IR_AHEAD_MM = 103.0        # IR sensors ahead of the car centre (official plate STL)
WORK_W = 640


def floor_level(hsv):
    """Median V and S of the frame border — the wood floor around the car."""
    border = np.r_[hsv[:10].reshape(-1, 3), hsv[-10:].reshape(-1, 3)]
    return np.median(border[:, 2]), np.median(border[:, 1])


def car_pose(hsv, prev_heading):
    """Everything that isn't floor (dark, saturated wires/boards, white ultrasonic), thin tape removed
    → (centre, heading unit vector, length px, hull). The battery box alone is the only uniformly
    dark part, so a darkness-only mask finds half the car."""
    fv, fs = floor_level(hsv)
    s, v = hsv[..., 1], hsv[..., 2]
    nonfloor = ((v < fv * 0.55) | (s > fs + 70) | ((v > 225) & (s < 35))).astype(np.uint8)
    body = cv2.morphologyEx(nonfloor, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    body = cv2.morphologyEx(body, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(body)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    pts = np.column_stack(np.nonzero(lab == i)[::-1]).astype(np.float32)
    hull = cv2.convexHull(pts)
    (cx, cy), (w, h), ang = cv2.minAreaRect(hull)
    length = max(w, h)
    theta = np.deg2rad(ang if w >= h else ang + 90)
    axis = np.array([np.cos(theta), np.sin(theta)])
    # front = end with the bright ultrasonic eyes; otherwise keep continuity with the last frame
    bright = (hsv[..., 2] > 200) & (hsv[..., 1] < 60)
    score = []
    for sgn in (1, -1):
        c = np.array([cx, cy]) + sgn * axis * length * 0.35
        x0, y0 = int(c[0]), int(c[1])
        r = int(length * 0.15)
        win = bright[max(0, y0 - r):y0 + r, max(0, x0 - r):x0 + r]
        score.append(win.mean() if win.size else 0)
    if prev_heading is not None and abs(score[0] - score[1]) < 0.01:
        axis = axis if axis @ prev_heading >= 0 else -axis
    elif score[1] > score[0]:
        axis = -axis
    return np.array([cx, cy]), axis, length, hull


def tape_lines(tape_mask, rng, max_lines=2, tol=4.0, strip=12.0):
    """RANSAC up to two straight tape legs from the visible tape pixels → [(point, unit dir, inliers)]."""
    pts = np.column_stack(np.nonzero(tape_mask)[::-1]).astype(float)
    lines = []
    for _ in range(max_lines):
        if len(pts) < 80:
            break
        best = None
        for _ in range(150):
            a, b = pts[rng.integers(len(pts), size=2)]
            d = b - a
            nrm = np.hypot(*d)
            if nrm < 20:
                continue
            d /= nrm
            dist = np.abs((pts - a) @ np.array([-d[1], d[0]]))
            inl = dist < tol
            if best is None or inl.sum() > best[2].sum():
                best = (a, d, inl)
        if best is None or best[2].sum() < 60:
            break
        a, d, inl = best
        vx, vy, x0, y0 = cv2.fitLine(pts[inl].astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        p0, d0 = np.array([x0, y0]), np.array([vx, vy])
        lines.append((p0, d0, pts[inl]))
        # drop the whole tape width (~21 px), not just the fitted centre band
        pts = pts[np.abs((pts - p0) @ np.array([-d0[1], d0[0]])) > strip]
    return lines


def measure(path, log=None, out_video=True):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer, rows, prev, rng = None, [], None, np.random.default_rng(0)
    k = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        scale = WORK_W / frame.shape[1]
        img = cv2.resize(frame, (WORK_W, int(frame.shape[0] * scale)))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        fv, _ = floor_level(hsv)
        dark = ((hsv[..., 2] < fv * 0.55) & (hsv[..., 1] < 120)).astype(np.uint8)
        t = k / fps
        k += 1
        pose = car_pose(hsv, prev)
        row = {"t_video": t, "found": 0}
        if pose is not None:
            centre, axis, length, hull = pose
            prev = axis
            mm = PLATE_MM / length
            car_mask = np.zeros_like(dark)
            cv2.fillConvexPoly(car_mask, hull.astype(np.int32), 1)
            car_mask = cv2.dilate(car_mask, np.ones((15, 15), np.uint8))
            tape = dark & (1 - car_mask)
            tape = cv2.morphologyEx(tape, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            lines = tape_lines(tape, rng)
            sensor = centre + axis * IR_AHEAD_MM / mm
            left = np.array([axis[1], -axis[0]])         # image y points down → car-left is this
            best = None
            for p, d, _ in lines:
                n = np.array([-d[1], d[0]])
                off = (sensor - p) @ n
                foot = sensor - n * off
                side = (foot - sensor) @ left             # + = tape on the car's left
                ang = np.degrees(np.arctan2(d @ left, d @ axis))
                ang = (ang + 90) % 180 - 90
                if best is None or abs(off) < abs(best[0]):
                    best = (off, side, ang, p, d)
            row.update(found=1, cx=centre[0], cy=centre[1], heading_deg=np.degrees(np.arctan2(axis[1], axis[0])),
                       mm_per_px=mm, n_lines=len(lines))
            if len(lines) == 2 and abs(lines[0][1] @ lines[1][1]) < 0.7:     # two legs ≥ ~45° apart
                (p1, d1, _), (p2, d2, _) = lines
                a, b = np.linalg.solve(np.column_stack([d1, -d2]), p2 - p1)
                corner = p1 + a * d1
                rel = corner - sensor
                row.update(corner_fwd_mm=(rel @ axis) * mm, corner_left_mm=(rel @ left) * mm)
            if best is not None:
                off, side, ang, p, d = best
                row.update(offset_mm=side * mm, angle_deg=ang)
            if out_video:
                cv2.polylines(img, [hull.astype(np.int32)], True, (0, 255, 255), 1)
                cv2.arrowedLine(img, tuple(centre.astype(int)), tuple((centre + axis * length * 0.5).astype(int)),
                                (0, 255, 0), 2, tipLength=0.2)
                cv2.circle(img, tuple(sensor.astype(int)), 4, (0, 0, 255), -1)
                for p, d, _ in lines:
                    a, b = p - d * 1000, p + d * 1000
                    cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), (255, 0, 255), 1)
                if "offset_mm" in row:
                    cv2.putText(img, f"t {t:5.2f}s  offset {row['offset_mm']:+5.0f} mm  angle {row['angle_deg']:+4.0f} deg",
                                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)
        if out_video:
            if writer is None:
                writer = cv2.VideoWriter(str(Path(path).with_suffix(".annotated.mp4")),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
            writer.write(img)
    cap.release()
    if writer:
        writer.release()

    keys = ["t_video", "found", "cx", "cy", "heading_deg", "mm_per_px", "n_lines", "offset_mm", "angle_deg",
            "corner_fwd_mm", "corner_left_mm"]
    out = Path(path).with_suffix(".track.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys + ["t_log"])
        w.writeheader()
        shift = sync(rows, log) if log else None
        for r in rows:
            if shift is not None:
                r["t_log"] = r["t_video"] - shift
            w.writerow({k: r.get(k, "") for k in keys + ["t_log"]})
    seen = [r for r in rows if "offset_mm" in r]
    print(f"{len(rows)} frames, car found in {sum(r['found'] for r in rows)}, tape measured in {len(seen)}")
    print(f"saved {out}" + ("" if not out_video else f" and {Path(path).with_suffix('.annotated.mp4')}"))
    return rows, shift


def sync(rows, log_path):
    """Video time of log t=0: maximise correlation of video tape side vs log IR centroid (R−L)."""
    with open(log_path) as f:
        lg = list(csv.DictReader(f))
    tl = np.array([float(r["t_s"]) for r in lg])
    act = np.array([[float(r["ir_act_l"]), float(r["ir_act_c"]), float(r["ir_act_r"])] for r in lg])
    ir = (act[:, 0] - act[:, 2]) / np.maximum(act.sum(1), 0.3)          # + = tape on the left
    tv = np.array([r["t_video"] for r in rows if "offset_mm" in r])
    ov = np.array([np.clip(r["offset_mm"] / 15.0, -1, 1) for r in rows if "offset_mm" in r])
    best = (-2, 0.0)
    for shift in np.arange(0, max(0.1, tv[-1] - tl[-1] + 5), 0.05):
        v = np.interp(tl + shift, tv, ov, left=np.nan, right=np.nan)
        ok = ~np.isnan(v)
        if ok.sum() < 100:
            continue
        c = np.corrcoef(v[ok], ir[ok])[0, 1]
        if c > best[0]:
            best = (c, shift)
    print(f"sync: log t=0 is video t={best[1]:.2f}s (correlation {best[0]:.2f})")
    return best[1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--log")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    measure(args.video, args.log, not args.no_video)
