"""Extract RETFound fundus image features and save one .pt file per image."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import models_vit


def prepare_model(checkpoint_path, device, arch="vit_large_patch16"):
    model = models_vit.__dict__[arch](
        img_size=224,
        num_classes=5,
        drop_path_rate=0,
        global_pool=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    return model.to(device).eval()


def prepare_image(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((224, 224))
        pixels = np.asarray(rgb) / 255.0

    # Normalize each channel using this image's own mean and standard deviation.
    mean = pixels.mean(axis=(0, 1), keepdims=True)
    std = pixels.std(axis=(0, 1), keepdims=True)
    pixels = (pixels - mean) / np.maximum(std, 1e-8)
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/zenglab/CIPHER/Fundus/RETFound_MAE-main/RETFound_cfp_weights.pth"),
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("/home/zenglab/CIPHER/Fundus/data_clean_new")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/home/zenglab/CIPHER/Fundus/embeddings")
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if not args.input_dir.is_dir():
        parser.error(f"Input directory not found: {args.input_dir}")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths = sorted(
        path for path in args.input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not paths:
        parser.error(f"No supported images found in {args.input_dir}")

    # Output filenames use the image stem, so reject collisions before saving.
    names = set()
    for path in paths:
        if path.stem in names:
            parser.error(f"Duplicate image stem would overwrite a feature file: {path.stem}")
        names.add(path.stem)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = prepare_model(args.checkpoint, device)
    print(f"Model loaded on {device}.")

    with torch.no_grad():
        for path in tqdm(paths, desc="Extracting features"):
            image = prepare_image(path).to(device)
            latent_feature = model.forward_features(image).squeeze(0)
            save_path = args.output_dir / f"{path.stem}.pt"
            torch.save(latent_feature.detach().cpu(), save_path)


if __name__ == "__main__":
    main()
