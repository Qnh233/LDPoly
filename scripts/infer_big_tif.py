#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from omegaconf import OmegaConf
import rasterio
from rasterio.windows import Window

from ldm.util import instantiate_from_config


def setup_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Use gloo for CPU tensor reduction at the end (more robust for this use case).
    dist.init_process_group(backend="gloo", init_method="env://")
    return True, rank, world_size, local_rank


def cleanup_distributed(is_distributed: bool):
    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def generate_starts(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def pad_to_size(chw: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    c, h, w = chw.shape
    if h == target_h and w == target_w:
        return chw
    out = np.zeros((c, target_h, target_w), dtype=chw.dtype)
    out[:, :h, :w] = chw
    return out


def to_uint8_rgb(arr: np.ndarray, fast_norm: bool = True) -> np.ndarray:
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


def load_model(config_path: str, checkpoint_path: str, device: torch.device, channels_last: bool = False):
    config = OmegaConf.load(config_path)
    config["model"]["target"] = "ldm.models.diffusion.ddpm_seg_vertex_inference.ExtendedLatentDiffusion"
    model = instantiate_from_config(config.model)

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[Checkpoint] missing={len(missing)}, unexpected={len(unexpected)}")

    model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model


@torch.no_grad()
def infer_batch(
    model,
    images_hwc: List[np.ndarray],
    names: List[str],
    sampler: str,
    ddim_steps: int,
    device: torch.device,
    amp: bool = False,
    amp_dtype: str = "fp16",
) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.stack(images_hwc, axis=0).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    images = torch.from_numpy(arr).to(device, non_blocking=True)

    bsz = images.shape[0]
    fake_seg = torch.zeros_like(images)
    fake_heat = torch.zeros_like(images)
    cls = torch.full((bsz, 1), -1, device=device, dtype=torch.long)

    model_batch = {
        "image": images,
        "segmentation": fake_seg,
        "heatmap": fake_heat,
        "file_path_": list(names),
        "class_id": cls,
    }

    amp_t = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    with torch.autocast(
        device_type="cuda" if device.type == "cuda" else "cpu",
        dtype=amp_t,
        enabled=(amp and device.type == "cuda"),
    ):
        outputs = model.log_images(
            model_batch,
            sampler=sampler,
            ddim_steps=ddim_steps,
            plot_denoise_rows=False,
            plot_diffusion_rows=False,
            return_first_stage_outputs=False,
            plot_conditioning_latent=False,
        )

    seg = outputs[f"samples_seg_{sampler}"]    # [B,C,H,W], [-1,1]
    heat = outputs[f"samples_heat_{sampler}"]  # [B,C,H,W], [-1,1]

    seg = (torch.clamp(seg, -1.0, 1.0) + 1.0) * 0.5
    heat = (torch.clamp(heat, -1.0, 1.0) + 1.0) * 0.5

    seg = seg.mean(dim=1).detach().cpu().numpy().astype(np.float32)   # [B,H,W]
    heat = heat.mean(dim=1).detach().cpu().numpy().astype(np.float32) # [B,H,W]
    return seg, heat


def reduce_sum_numpy(arr: np.ndarray, rank: int, world_size: int, chunk_rows: int = 2048):
    if world_size == 1:
        return arr

    h = arr.shape[0]
    out = np.zeros_like(arr) if rank == 0 else None
    for s in range(0, h, chunk_rows):
        e = min(s + chunk_rows, h)
        t = torch.from_numpy(arr[s:e]).contiguous()  # CPU tensor
        dist.reduce(t, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            out[s:e] = t.numpy()
    return out


def write_geotiff(out_path: str, src_profile: dict, arr2d: np.ndarray, compress: str = "lzw"):
    profile = src_profile.copy()
    profile.update(
        driver="GTiff",
        height=arr2d.shape[0],
        width=arr2d.shape[1],
        count=1,
        dtype=arr2d.dtype,
        compress=compress,
        bigtiff="if_safer",
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr2d, 1)


def main():
    parser = argparse.ArgumentParser("Multi-GPU big TIF sliding-window inference and stitching")
    parser.add_argument("--input_tif", type=str, required=True)
    parser.add_argument("--out_seg_tif", type=str, required=True)
    parser.add_argument("--out_heat_tif", type=str, default="")
    parser.add_argument("--run_name", type=str, required=True, help="logs/<run_name>")
    parser.add_argument("--model_ckpt", type=str, required=True, help="checkpoint under logs/<run_name>/checkpoints")
    parser.add_argument("--sampler", type=str, default="ddim", choices=["direct", "ddim", "ddpm"])
    parser.add_argument("--ddim_steps", type=int, default=20)

    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--edge_mode", type=str, default="pad", choices=["pad", "drop"])
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--merge", type=str, default="mean", choices=["mean", "max", "overwrite"])
    parser.add_argument("--binary_thr", type=float, default=-1.0, help=">=0 to binarize seg output")
    parser.add_argument("--heat_binary_thr", type=float, default=-1.0, help=">=0 to binarize heat output")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compress", type=str, default="lzw", choices=["lzw", "deflate", "none"])

    parser.add_argument("--amp", action="store_true", help="enable mixed precision inference")
    parser.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--channels_last", action="store_true")
    parser.add_argument("--reduce_chunk_rows", type=int, default=2048, help="row chunk for distributed reduce")
    args = parser.parse_args()

    if args.tile_size <= 0 or args.stride <= 0 or args.batch_size <= 0:
        raise ValueError("tile_size/stride/batch_size must be > 0")

    is_distributed, rank, world_size, local_rank = setup_distributed()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if is_distributed else "cuda:0")
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    if rank == 0:
        print(f"[Stage] device = {device}, world_size={world_size}")

    config_candidates = glob.glob(os.path.join("logs", args.run_name, "configs", "*-project.yaml"))
    if len(config_candidates) == 0:
        raise FileNotFoundError(f"No config found under logs/{args.run_name}/configs/*-project.yaml")
    config_path = config_candidates[0]
    ckpt_path = os.path.join("logs", args.run_name, "checkpoints", args.model_ckpt)

    if rank == 0:
        print("[Stage] loading model")
    model = load_model(config_path, ckpt_path, device, channels_last=args.channels_last)

    with rasterio.open(args.input_tif) as src:
        H, W = src.height, src.width
        y_starts = generate_starts(H, args.tile_size, args.stride)
        x_starts = generate_starts(W, args.tile_size, args.stride)
        windows = [(y, x) for y in y_starts for x in x_starts]

        # contiguous split -> better disk locality than round-robin
        idx_chunks = np.array_split(np.arange(len(windows)), world_size)
        local_idx = idx_chunks[rank]
        local_windows = [windows[i] for i in local_idx]

        if rank == 0:
            print(f"[Stage] total windows={len(windows)} (H={H}, W={W}), local={len(local_windows)} per-rank approx")

        seg_sum = np.zeros((H, W), dtype=np.float32)
        heat_sum = np.zeros((H, W), dtype=np.float32)
        cnt = np.zeros((H, W), dtype=np.float32)

        pbar = tqdm(total=len(local_windows), desc=f"Sliding infer rank{rank}", disable=(rank != 0))
        batch_imgs = []
        batch_meta = []
        batch_names = []

        def flush_batch():
            nonlocal batch_imgs, batch_meta, batch_names, seg_sum, heat_sum, cnt
            if len(batch_imgs) == 0:
                return

            seg_pred, heat_pred = infer_batch(
                model=model,
                images_hwc=batch_imgs,
                names=batch_names,
                sampler=args.sampler,
                ddim_steps=args.ddim_steps,
                device=device,
                amp=args.amp,
                amp_dtype=args.amp_dtype,
            )

            for i, (y, x, read_h, read_w) in enumerate(batch_meta):
                tile_seg = seg_pred[i][:read_h, :read_w]
                tile_heat = heat_pred[i][:read_h, :read_w]

                if args.merge == "mean":
                    seg_sum[y:y + read_h, x:x + read_w] += tile_seg
                    heat_sum[y:y + read_h, x:x + read_w] += tile_heat
                    cnt[y:y + read_h, x:x + read_w] += 1.0
                elif args.merge == "max":
                    # for max/overwrite we still reduce later with SUM logic, so convert to sum/count here
                    seg_sum[y:y + read_h, x:x + read_w] += tile_seg
                    heat_sum[y:y + read_h, x:x + read_w] += tile_heat
                    cnt[y:y + read_h, x:x + read_w] += 1.0
                else:
                    seg_sum[y:y + read_h, x:x + read_w] += tile_seg
                    heat_sum[y:y + read_h, x:x + read_w] += tile_heat
                    cnt[y:y + read_h, x:x + read_w] += 1.0

            batch_imgs = []
            batch_meta = []
            batch_names = []

        for y, x in local_windows:
            read_h = min(args.tile_size, H - y)
            read_w = min(args.tile_size, W - x)

            if args.edge_mode == "drop" and (read_h < args.tile_size or read_w < args.tile_size):
                pbar.update(1)
                continue

            win = Window(col_off=x, row_off=y, width=read_w, height=read_h)
            chw = src.read(window=win)
            if chw.shape[0] == 0:
                pbar.update(1)
                continue

            if args.edge_mode == "pad":
                chw = pad_to_size(chw, args.tile_size, args.tile_size)

            rgb = to_uint8_rgb(chw, fast_norm=True)
            batch_imgs.append(rgb)
            batch_meta.append((y, x, read_h, read_w))
            batch_names.append(f"y{y}_x{x}")

            if len(batch_imgs) >= args.batch_size:
                flush_batch()
            pbar.update(1)

        flush_batch()
        pbar.close()

        if rank == 0:
            print("[Stage] reducing seg_sum / heat_sum / count across ranks")
        seg_sum_g = reduce_sum_numpy(seg_sum, rank, world_size, chunk_rows=args.reduce_chunk_rows)
        heat_sum_g = reduce_sum_numpy(heat_sum, rank, world_size, chunk_rows=args.reduce_chunk_rows)
        cnt_g = reduce_sum_numpy(cnt, rank, world_size, chunk_rows=args.reduce_chunk_rows)

        if rank == 0:
            print("[Stage] merging outputs")
            valid = cnt_g > 0
            seg_canvas = np.zeros_like(seg_sum_g, dtype=np.float32)
            heat_canvas = np.zeros_like(heat_sum_g, dtype=np.float32)
            seg_canvas[valid] = seg_sum_g[valid] / cnt_g[valid]
            heat_canvas[valid] = heat_sum_g[valid] / cnt_g[valid]

            if args.binary_thr >= 0:
                seg_out = (seg_canvas >= args.binary_thr).astype(np.uint8) * 255
            else:
                seg_out = np.clip(seg_canvas * 255.0, 0, 255).astype(np.uint8)

            if args.out_heat_tif:
                if args.heat_binary_thr >= 0:
                    heat_out = (heat_canvas >= args.heat_binary_thr).astype(np.uint8) * 255
                else:
                    heat_out = np.clip(heat_canvas * 255.0, 0, 255).astype(np.uint8)
            else:
                heat_out = None

            print("[Stage] writing GeoTIFF")
            os.makedirs(os.path.dirname(args.out_seg_tif) or ".", exist_ok=True)
            comp = None if args.compress == "none" else args.compress
            write_geotiff(args.out_seg_tif, src.profile, seg_out, compress=comp)
            print(f"[Done] seg tif: {args.out_seg_tif}")

            if args.out_heat_tif and heat_out is not None:
                os.makedirs(os.path.dirname(args.out_heat_tif) or ".", exist_ok=True)
                write_geotiff(args.out_heat_tif, src.profile, heat_out, compress=comp)
                print(f"[Done] heat tif: {args.out_heat_tif}")

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
