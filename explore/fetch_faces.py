"""Backfill original HEIC faces for (pano, face) pairs a detection run already processed, so every crop has its
untouched source image under data/<city>/apple/faces/<pano>_<face>.heic.

Usage: uv run explore/fetch_faces.py coruna store_sign [--workers 16]
"""
import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from streetlevel import lookaround


class TimeoutSession(requests.Session):
    def get(self, *a, **kw):
        kw.setdefault("timeout", 30)
        return super().get(*a, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("slug", nargs="?", default="store_sign")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    data = Path("data") / a.city
    faces = data / "apple" / "faces"
    faces.mkdir(parents=True, exist_ok=True)
    build = {r["id"]: r["build_id"] for r in csv.DictReader(open(data / "lookaround.csv"))}
    todo = []
    for line in (data / "apple" / a.slug / "done.txt").read_text().split():
        pano, face = line.rsplit("_", 1)
        if not (faces / f"{pano}_{face}.heic").exists():
            todo.append((pano, int(face)))
    print(len(todo), "faces to fetch")
    auth = lookaround.Authenticator()

    def one(job):
        pano, face = job
        s = TimeoutSession()
        for _ in range(3):
            try:
                data_ = lookaround.get_panorama_face((int(pano), int(build[pano])), face, 0, auth, session=s)
                (faces / f"{pano}_{face}.heic").write_bytes(data_)
                return 1
            except Exception:
                time.sleep(1)
        print("failed", pano, face, flush=True)
        return 0

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        ok = sum(ex.map(one, todo))
    print(f"{ok}/{len(todo)} fetched in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
