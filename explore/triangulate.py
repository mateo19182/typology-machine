"""Put every grouped sign on the map: each view of a sign is a ray from the pano position toward the sign
(bearing from the pixel position in the face or view). Intersect the rays of a group in least squares, drop
outliers, and write sign_positions.jsonl: sign, lat, lon, n_rays, residual_m, height_m (from the vertical
angle and the camera height). Signs seen from one point only get a position 8 m along the ray, flagged.

Apple faces and Google perspective views have different geometry; both are handled here.
Usage: uv run explore/triangulate.py coruna store_sign --source apple
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

CAM_H = {"apple": 2.4, "gsv": 2.5}  # metres, car roof rigs
# Apple side faces: equirectangular patches. fov_s (phi), fov_h (theta), cy offset; yaw per face.
APPLE_FACE = {0: (0.0, 2.159844949), 1: (math.pi / 2, 1.178097245), 2: (math.pi, 2.159844949), 3: (-math.pi / 2, 1.178097245)}
APPLE_FOV_H, APPLE_CY = 1.832595715, 0.305432619


def ray(r, source):
    """returns (bearing_rad clockwise from north, elevation_rad)"""
    x0, y0, x1, y1 = r["box"]
    if source == "apple":
        W, H = r["face_size"]
        u, v = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
        yaw, fov_s = APPLE_FACE[r["face"]]
        phi = r["heading"] + yaw + (u - 0.5) * fov_s  # heading in radians
        # theta measured from zenith in the stitched equirect: theta = pi/2 - fov_h/2 - cy + v*fov_h
        theta = math.pi / 2 - APPLE_FOV_H / 2 - APPLE_CY + v * APPLE_FOV_H
        return phi % (2 * math.pi), math.pi / 2 - theta
    else:
        W, H = r["view_size"]
        u, v = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
        fh, fv = (math.radians(f) for f in r["fov"])
        f = 0.5 / math.tan(fh / 2)
        rel = math.atan((u - 0.5) / f)
        fy = 0.5 / math.tan(fv / 2)
        pitch = math.radians(r["view_pitch"]) + math.atan((0.5 - v) / fy)
        return math.radians((r["heading"] or 0) + r["view_yaw"]) + rel, pitch


def intersect(points, dirs):
    """least-squares point closest to all rays (2D). points: n x 2, dirs: n x 2 unit."""
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for p, d in zip(points, dirs):
        P = np.eye(2) - np.outer(d, d)
        A += P
        b += P @ p
    return np.linalg.lstsq(A, b, rcond=None)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("slug", nargs="?", default="store_sign")
    ap.add_argument("--source", default="apple")
    ap.add_argument("--max-res", type=float, default=3.0, help="drop rays farther than this (m) from the solution")
    a = ap.parse_args()
    base = Path("data") / a.city / a.source / a.slug
    rows = {r["id"]: r for r in map(json.loads, open(base / "instances.jsonl"))}
    groups = [json.loads(l) for l in open(base / "groups.jsonl")]
    lat0 = next(iter(rows.values()))["lat"]
    kx, ky = 111320 * math.cos(math.radians(lat0)), 111320

    def xy(r):
        return np.array([r["lon"] * kx, r["lat"] * ky])

    out = open(base / "sign_positions.jsonl", "w")
    stats = {"multi": 0, "single": 0, "failed": 0}
    for g in groups:
        members = [rows[m] for m in g["members"] if m in rows]
        rays = []
        for r in members:
            try:
                b, e = ray(r, a.source)
            except KeyError:
                continue
            rays.append((xy(r), np.array([math.sin(b), math.cos(b)]), e))  # x east, y north
        if not rays:
            stats["failed"] += 1
            continue
        if len(rays) == 1:
            p, d, e = rays[0]
            pos = p + 8 * d
            res, n, dist = None, 1, 8.0
            stats["single"] += 1
        else:
            pts = np.array([r[0] for r in rays])
            dirs = np.array([r[1] for r in rays])
            pos = intersect(pts, dirs)
            # outlier pass: distance from point to each ray (only forward)
            keep = []
            for p, d, e in rays:
                t = (pos - p) @ d
                perp = np.linalg.norm((pos - p) - t * d)
                if t > 1 and perp <= a.max_res:
                    keep.append((p, d, e))
            if len(keep) >= 2:
                pos = intersect(np.array([k[0] for k in keep]), np.array([k[1] for k in keep]))
                rays = keep
            n = len(rays)
            res = float(np.mean([np.linalg.norm((pos - p) - ((pos - p) @ d) * d) for p, d, e in rays]))
            dist = float(np.mean([np.linalg.norm(pos - p) for p, d, e in rays]))
            stats["multi"] += 1
        height = CAM_H[a.source] + float(np.mean([math.tan(e) * np.linalg.norm(pos - p) for p, d, e in rays]))
        out.write(json.dumps({"sign": g.get("sign"), "rep": g["rep"], "lat": round(float(pos[1] / ky), 7), "lon": round(float(pos[0] / kx), 7),
                              "n_rays": n, "residual_m": None if res is None else round(res, 2), "dist_m": round(dist, 1),
                              "height_m": round(height, 1)}) + "\n")
    print(stats, "->", base / "sign_positions.jsonl")


if __name__ == "__main__":
    main()
