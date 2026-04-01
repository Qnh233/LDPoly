#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse

import cv2
import numpy as np
import rasterio
from tqdm import tqdm


def infer_canvas_size(tiles):
    max_x = 0
    max_y = 0
    for t in tiles:
        x = int(t["x"])
        y = int(t["y"])
        rw = int(t["read_w"])
        rh = int(t["read_h"])
        max_x = max(max_x, x + rw)
        max_y = max(max_y, y + rh)
    return max_y, max_x  # H, W


def load_tile(path, pred_ext):
    if pred_ext == "npy":
        tile = np.load(path).astype(np.float32)
        if tile.ndim == 3:
            tile = tile[:, :, 0]
        return tile
    # png mode
    tile = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if tile is None:
        return None
    return (tile.astype(np.float32) / 255.0)


def main():
    parser = argparse.ArgumentParser("Stitch tile predictions back to large map")
    parser.add_argument("--manifest", type=str, required=True, help="tiles_manifest.json from tif_to_tiles.py")
    parser.add_argument("--tile_pred_dir", type=str, required=True, help="e.g. out/samples_seg_ddim")
    parser.add_argument("--pred_ext", type=str, default="png", choices=["png", "npy"], help="Tile prediction format")
    parser.add_argument("--out_tif", type=str, required=True, help="stitched output .tif")
    parser.add_argument("--merge", type=str, default="mean", choices=["mean", "max", "overwrite"])
    parser.add_argument("--binary_thr", type=float, default=-1.0, help=">=0 to save binary map")
    parser.add_argument("--strict_missing", action="store_true", help="Fail if any tile is missing")
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    tiles = manifest["tiles"]
    H, W = infer_canvas_size(tiles)

    if args.merge == "max":
        canvas = np.full((H, W), -np.inf, dtype=np.float32)
    else:
        canvas = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    missing = 0
    for t in tqdm(tiles, desc="Stitching"):
        stem = os.path.splitext(t["file_name"])[0]
        p = os.path.join(args.tile_pred_dir, f"{stem}.{args.pred_ext}")

        if not os.path.exists(p):
            missing += 1
            continue

        tile = load_tile(p, args.pred_ext)
        if tile is None:
            missing += 1
            continue

        x = int(t["x"])
        y = int(t["y"])
        rw = int(t["read_w"])
        rh = int(t["read_h"])

        tile = tile[:rh, :rw]

        if args.merge == "mean":
            canvas[y:y + rh, x:x + rw] += tile
            count[y:y + rh, x:x + rw] += 1.0
        elif args.merge == "max":
            canvas[y:y + rh, x:x + rw] = np.maximum(canvas[y:y + rh, x:x + rw], tile)
            count[y:y + rh, x:x + rw] = 1.0
        else:  # overwrite
            canvas[y:y + rh, x:x + rw] = tile
            count[y:y + rh, x:x + rw] = 1.0

    valid = count > 0
    if args.merge == "mean":
        canvas[valid] = canvas[valid] / count[valid]
        canvas[~valid] = 0.0
    elif args.merge == "max":
        canvas[~valid] = 0.0

    if missing > 0:
        miss_ratio = missing / max(len(tiles), 1)
        print(f"[Warn] missing tiles: {missing}/{len(tiles)} ({miss_ratio:.2%})")
        if args.strict_missing:
            raise RuntimeError("Missing tiles detected under --strict_missing")

    out_dir = os.path.dirname(args.out_tif)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print("[Stage] merge done")
    print("[Stage] build out_arr start")
    if args.binary_thr >= 0:
        out_arr = (canvas >= args.binary_thr).astype(np.uint8) * 255
    else:
        out_arr = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
    print("[Stage] build out_arr done")
    print("[Stage] write tif start")
    src_path = manifest.get("input", None)
    if src_path is not None and os.path.exists(src_path):
        with rasterio.open(src_path) as src:
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                height=out_arr.shape[0],
                width=out_arr.shape[1],
                count=1,
                dtype=out_arr.dtype,
                compress="deflate",
                predictor=2,
                bigtiff="if_safer",
            )
            with rasterio.open(args.out_tif, "w", **profile) as dst:
                dst.write(out_arr, 1)
    else:
        # Fallback: no georef info
        profile = {
            "driver": "GTiff",
            "height": out_arr.shape[0],
            "width": out_arr.shape[1],
            "count": 1,
            "dtype": out_arr.dtype,
            "compress": "deflate",
        }
        with rasterio.open(args.out_tif, "w", **profile) as dst:
            dst.write(out_arr, 1)
    print("[Stage] write tif done")
    print(f"[Done] stitched shape: {canvas.shape}")
    print(f"[Done] tif: {args.out_tif}")


if __name__ == "__main__":
    main()
