#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import argparse
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config


class TileDataset(Dataset):
    def __init__(self, input_dir, image_size=(256, 256), exts=None):
        self.input_dir = input_dir
        self.image_size = image_size
        if exts is None:
            exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

        self.paths = []
        for e in exts:
            self.paths.extend(glob.glob(os.path.join(input_dir, f"*{e}")))
            self.paths.extend(glob.glob(os.path.join(input_dir, f"*{e.upper()}")))

        self.paths = sorted(set(self.paths))
        if len(self.paths) == 0:
            raise ValueError(f"No images found in {input_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        name = os.path.splitext(os.path.basename(p))[0]
        img = Image.open(p).convert("RGB")
        img = img.resize(self.image_size, Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0  # [-1,1]
        return {
            "image": torch.from_numpy(arr),  # [H,W,3]
            "name": name,
        }


def setup_distributed():
    """Initialize torch.distributed if launched with torchrun."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return True, rank, world_size, local_rank


def cleanup_distributed(is_distributed):
    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def load_model(config_path, checkpoint_path, device):
    config = OmegaConf.load(config_path)
    config["model"]["target"] = "ldm.models.diffusion.ddpm_seg_vertex_inference.ExtendedLatentDiffusion"
    model = instantiate_from_config(config.model)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[Checkpoint] missing={len(missing)}, unexpected={len(unexpected)}")

    model.to(device)
    model.eval()
    return model


def save_one(seg_map, heat_map, name, outdir, sampler):
    # only PNG outputs, no NPY
    seg_png_dir = os.path.join(outdir, f"samples_seg_{sampler}")
    heat_png_dir = os.path.join(outdir, f"samples_heat_{sampler}")
    os.makedirs(seg_png_dir, exist_ok=True)
    os.makedirs(heat_png_dir, exist_ok=True)

    cv2.imwrite(
        os.path.join(seg_png_dir, f"{name}.png"),
        (seg_map * 255.0).astype(np.uint8)
    )
    cv2.imwrite(
        os.path.join(heat_png_dir, f"{name}.png"),
        (heat_map * 255.0).astype(np.uint8)
    )


@torch.no_grad()
def run_batched_inference(
    model,
    loader,
    outdir,
    sampler="ddim",
    ddim_steps=20,
    device="cuda",
    io_threads=8,
    rank=0,
):
    pool = ThreadPoolExecutor(max_workers=io_threads)
    futures = []

    pbar = tqdm(loader, desc=f"Batched inference [rank {rank}]", disable=(rank != 0))
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)  # [B,H,W,3]
        names = batch["name"]
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

        outputs = model.log_images(
            model_batch,
            sampler=sampler,
            ddim_steps=ddim_steps,
            plot_denoise_rows=False,
            plot_diffusion_rows=False,
            return_first_stage_outputs=False,
            plot_conditioning_latent=False,
        )

        seg = outputs[f"samples_seg_{sampler}"]   # [B,C,H,W], [-1,1]
        heat = outputs[f"samples_heat_{sampler}"] # [B,C,H,W], [-1,1]

        seg = torch.clamp(seg, -1.0, 1.0)
        heat = torch.clamp(heat, -1.0, 1.0)

        seg = (seg + 1.0) / 2.0
        heat = (heat + 1.0) / 2.0

        seg = seg.mean(dim=1).detach().cpu().numpy()   # [B,H,W]
        heat = heat.mean(dim=1).detach().cpu().numpy() # [B,H,W]

        for i, n in enumerate(names):
            futures.append(pool.submit(save_one, seg[i], heat[i], n, outdir, sampler))

    # ensure all writes complete
    for f in tqdm(futures, desc=f"Saving outputs [rank {rank}]", disable=(rank != 0)):
        f.result()
    pool.shutdown()


def main():
    parser = argparse.ArgumentParser("Batched tile inference for LDPoly (multi-GPU DDP)")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True, help="logs/<run>")
    parser.add_argument("--model_ckpt", type=str, required=True, help="checkpoint filename under logs/<run>/checkpoints")
    parser.add_argument("--sampler", type=str, default="ddim", choices=["direct", "ddim", "ddpm"])
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8, help="Per-GPU batch size")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--io_threads", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256])
    args = parser.parse_args()

    is_distributed, rank, world_size, local_rank = setup_distributed()

    if args.device == "cuda" and torch.cuda.is_available():
        if is_distributed:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    config_path = glob.glob(os.path.join("logs", args.run_name, "configs", "*-project.yaml"))[0]
    ckpt_path = os.path.join("logs", args.run_name, "checkpoints", args.model_ckpt)

    model = load_model(config_path, ckpt_path, device)

    ds = TileDataset(args.input_dir, image_size=tuple(args.image_size))
    sampler = None
    if is_distributed:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    run_batched_inference(
        model=model,
        loader=loader,
        outdir=args.outdir,
        sampler=args.sampler,
        ddim_steps=args.ddim_steps,
        device=device,
        io_threads=args.io_threads,
        rank=rank,
    )

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
