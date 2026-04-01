#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Slice a large TIF/GeoTIFF into fixed-size tiles (e.g., 256x256) for LDPoly inference.

Output:
- tiled PNG images in output_dir
- tiles_manifest.json with source pixel offsets/sizes

Speed-oriented features:
- threaded PNG writing
- faster blank-tile check via bincount + optional downsample
- optional fast normalization for float images

Example:
    python scripts/tif_to_tiles.py \
      --input /data/big_map.tif \
      --output_dir /data/big_map_tiles \
      --tile_size 256 \
      --stride 256 \
      --edge_mode pad \
      --io_workers 8 \
      --fast_norm
"""

import os
import json
import argparse
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import rasterio
    from rasterio.windows import Window
except Exception as e:
    raise ImportError(
        "This script requires rasterio. Install with: pip install rasterio"
    ) from e


def to_uint8_rgb(arr: np.ndarray, fast_norm: bool = False) -> np.ndarray:
    """
    Convert input tile array to uint8 RGB.
    Input arr: (C, H, W), where C can be 1+.
    """
    c, h, w = arr.shape

    if c >= 3:
        arr = arr[:3, :, :]
    elif c == 1:
        arr = np.repeat(arr, 3, axis=0)
    else:
        arr = np.zeros((3, h, w), dtype=np.uint8)

    if arr.dtype == np.uint8:
        out = arr
    elif arr.dtype == np.uint16:
        out = (arr / 257.0).clip(0, 255).astype(np.uint8)
    elif np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0 and arr.min() >= 0.0:
            out = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            out = np.empty_like(arr, dtype=np.uint8)
            for i in range(arr.shape[0]):
                ch = arr[i]
                if fast_norm:
                    lo = float(ch.min())
                    hi = float(ch.max())
                else:
                    lo = float(np.percentile(ch, 2))
                    hi = float(np.percentile(ch, 98))
                if hi <= lo:
                    out[i] = np.zeros_like(ch, dtype=np.uint8)
                else:
                    out[i] = ((ch - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.float32)
        mn, mx = float(arr.min()), float(arr.max())
        if mx <= mn:
            out = np.zeros_like(arr, dtype=np.uint8)
        else:
            out = ((arr - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)

    return np.transpose(out, (1, 2, 0))  # CHW -> HWC


def pad_to_size(chw: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Pad (C,H,W) to target size with zeros.
    """
    c, h, w = chw.shape
    if h == target_h and w == target_w:
        return chw
    padded = np.zeros((c, target_h, target_w), dtype=chw.dtype)
    padded[:, :h, :w] = chw
    return padded


def is_mostly_blank(
    rgb: np.ndarray,
    blank_ratio_thr: float = 0.995,
    sample_step: int = 4,
) -> bool:
    """
    Fast blank-tile filter:
    - Optional spatial downsample by step
    - Dominant grayscale bin ratio via bincount
    """
    if sample_step > 1:
        rgb = rgb[::sample_step, ::sample_step, :]

    gray = (
        rgb[..., 0].astype(np.uint16)
        + rgb[..., 1].astype(np.uint16)
        + rgb[..., 2].astype(np.uint16)
    ) // 3
    gray = gray.astype(np.uint8)

    hist = np.bincount(gray.ravel(), minlength=256)
    dominant = hist.max() / max(gray.size, 1)
    return dominant >= blank_ratio_thr


def generate_starts(length: int, tile: int, stride: int) -> List[int]:
    """
    Generate start indices; ensure last tile can cover the tail when edge_mode=pad.
    """
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def save_png(path: str, rgb: np.ndarray, compress_level: int) -> None:
    """
    Save PNG with configurable compression level.
    compress_level: 0 (fastest, large file) .. 9 (slowest, small file)
    """
    img = Image.fromarray(rgb)
    img.save(path, compress_level=compress_level)


def flush_futures(futures):
    for f in futures:
        f.result()
    futures.clear()


def main():
    parser = argparse.ArgumentParser(description="Slice large TIF into fixed-size tiles.")
    parser.add_argument("--input", type=str, required=True, help="Input .tif/.tiff file path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save tiles")
    parser.add_argument("--tile_size", type=int, default=256, help="Tile size (square)")
    parser.add_argument("--stride", type=int, default=256, help="Stride in pixels")
    parser.add_argument(
        "--edge_mode",
        type=str,
        default="pad",
        choices=["pad", "drop"],
        help="How to handle border tiles smaller than tile_size",
    )
    parser.add_argument(
        "--skip_blank",
        action="store_true",
        help="Skip mostly blank tiles",
    )
    parser.add_argument(
        "--blank_ratio_thr",
        type=float,
        default=0.995,
        help="Dominant grayscale ratio threshold for blank filtering",
    )
    parser.add_argument(
        "--blank_sample_step",
        type=int,
        default=4,
        help="Downsample step for blank check. Larger is faster.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="tile",
        help="Output filename prefix",
    )
    parser.add_argument(
        "--io_workers",
        type=int,
        default=8,
        help="Number of threads used for PNG writing.",
    )
    parser.add_argument(
        "--fast_norm",
        action="store_true",
        help="Use min-max instead of percentile normalization for float tiles.",
    )
    parser.add_argument(
        "--png_compress_level",
        type=int,
        default=1,
        help="PNG compression level [0..9]. Lower is faster.",
    )

    args = parser.parse_args()

    if args.tile_size <= 0 or args.stride <= 0:
        raise ValueError("tile_size and stride must be > 0")
    if not (0 <= args.png_compress_level <= 9):
        raise ValueError("png_compress_level must be in [0, 9]")
    if args.blank_sample_step <= 0:
        raise ValueError("blank_sample_step must be > 0")
    if args.io_workers <= 0:
        raise ValueError("io_workers must be > 0")

    os.makedirs(args.output_dir, exist_ok=True)

    manifest: Dict[str, object] = {
        "input": os.path.abspath(args.input),
        "tile_size": args.tile_size,
        "stride": args.stride,
        "edge_mode": args.edge_mode,
        "tiles": [],
    }

    with rasterio.open(args.input) as ds:
        h, w = ds.height, ds.width
        bands = ds.count

        y_starts = generate_starts(h, args.tile_size, args.stride)
        x_starts = generate_starts(w, args.tile_size, args.stride)

        total = len(y_starts) * len(x_starts)
        print(f"[Info] Input size: {w}x{h}, bands={bands}, candidate tiles={total}")

        pbar = tqdm(total=total, desc="Slicing TIF")
        tile_idx = 0
        futures = []

        with ThreadPoolExecutor(max_workers=args.io_workers) as executor:
            for y in y_starts:
                for x in x_starts:
                    read_h = min(args.tile_size, h - y)
                    read_w = min(args.tile_size, w - x)

                    if args.edge_mode == "drop" and (read_h < args.tile_size or read_w < args.tile_size):
                        pbar.update(1)
                        continue

                    window = Window(col_off=x, row_off=y, width=read_w, height=read_h)
                    chw = ds.read(window=window)  # (C,H,W)

                    if chw.shape[0] == 0:
                        pbar.update(1)
                        continue

                    if args.edge_mode == "pad":
                        chw = pad_to_size(chw, args.tile_size, args.tile_size)

                    rgb = to_uint8_rgb(chw, fast_norm=args.fast_norm)

                    if args.skip_blank and is_mostly_blank(
                        rgb,
                        blank_ratio_thr=args.blank_ratio_thr,
                        sample_step=args.blank_sample_step,
                    ):
                        pbar.update(1)
                        continue

                    name = f"{args.prefix}_y{y:06d}_x{x:06d}.png"
                    save_path = os.path.join(args.output_dir, name)

                    futures.append(
                        executor.submit(
                            save_png,
                            save_path,
                            rgb,
                            args.png_compress_level,
                        )
                    )

                    # Avoid unbounded future growth
                    if len(futures) >= args.io_workers * 4:
                        flush_futures(futures)

                    manifest["tiles"].append({
                        "file_name": name,
                        "x": int(x),
                        "y": int(y),
                        "read_w": int(read_w),
                        "read_h": int(read_h),
                        "tile_w": int(args.tile_size),
                        "tile_h": int(args.tile_size),
                        "bands": int(min(3, bands)) if bands > 0 else 0,
                    })

                    tile_idx += 1
                    pbar.update(1)

            if futures:
                flush_futures(futures)

        pbar.close()

    manifest_path = os.path.join(args.output_dir, "tiles_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Done] Saved {tile_idx} tiles to: {args.output_dir}")
    print(f"[Done] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
