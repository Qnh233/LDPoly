#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse

import cv2
import numpy as np
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


def main():
    parser = argparse.ArgumentParser("Stitch tile logits back to large map")
    parser.add_argument("--manifest", type=str, required=True, help="tiles_manifest.json from tif_to_tiles.py")
    parser.add_argument("--tile_logits_dir", type=str, required=True, help="e.g. out/samples_seg_ddim_logits_npy")
    parser.add_argument("--out_npy", type=str, required=True, help="stitched float map .npy")
    parser.add_argument("--out_png", type=str, required=True, help="stitched preview .png")
    parser.add_argument("--merge", type=str, default="mean", choices=["mean", "max", "overwrite"])
    parser.add_argument("--binary_thr", type=float, default=-1.0, help=">=0 to also save binary map in out_png")
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
        name = t["file_name"]
        stem = os.path.splitext(name)[0]
        p = os.path.join(args.tile_logits_dir, f"{stem}.npy")
        if not os.path.exists(p):
            missing += 1
            continue

        tile = np.load(p).astype(np.float32)  # [tile_h, tile_w] or [H,W,C]
        if tile.ndim == 3:
            tile = tile[:, :, 0]

        x = int(t["x"])
        y = int(t["y"])
        rw = int(t["read_w"])
        rh = int(t["read_h"])

        tile = tile[:rh, :rw]

        if args.merge == "mean":
            canvas[y:y+rh, x:x+rw] += tile
            count[y:y+rh, x:x+rw] += 1.0
        elif args.merge == "max":
            canvas[y:y+rh, x:x+rw] = np.maximum(canvas[y:y+rh, x:x+rw], tile)
            count[y:y+rh, x:x+rw] = 1.0
        else:  # overwrite
            canvas[y:y+rh, x:x+rw] = tile
            count[y:y+rh, x:x+rw] = 1.0

    valid = count > 0
    if args.merge == "mean":
        canvas[valid] = canvas[valid] / count[valid]
        canvas[~valid] = 0.0
    elif args.merge == "max":
        canvas[~valid] = 0.0

    os.makedirs(os.path.dirname(args.out_npy), exist_ok=True)
    np.save(args.out_npy, canvas.astype(np.float32))

    if args.binary_thr >= 0:
        vis = (canvas >= args.binary_thr).astype(np.uint8) * 255
    else:
        vis = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(args.out_png, vis)

    print(f"[Done] stitched shape: {canvas.shape}, missing tiles: {missing}")
    print(f"[Done] npy: {args.out_npy}")
    print(f"[Done] png: {args.out_png}")


if __name__ == "__main__":
    main()
