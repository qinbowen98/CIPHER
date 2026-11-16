"""Extract and pair RETFound embeddings, or pair existing per-image .pt files."""

import argparse
import inspect
import re
import sys
from pathlib import Path

import torch


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PAIR_STEM = re.compile(r"(?P<sample>.+)\.(?P<side>[01])_1")


def load_tensor_file(path):
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = True
    return torch.load(path, **options)


def eye_vector(value, name):
    if isinstance(value, dict) and "feat" in value:
        value = value["feat"]
    if not torch.is_tensor(value):
        raise ValueError(f"{name}: expected a tensor or a dictionary containing a 'feat' tensor.")
    vector = value.detach().cpu().squeeze()
    if vector.shape != (1024,):
        raise ValueError(f"{name}: expected a 1024-value vector, got shape {tuple(value.shape)}.")
    if not torch.isfinite(vector).all().item():
        raise ValueError(f"{name}: the embedding contains nonfinite values.")
    return vector


def find_pairs(directory, extensions):
    samples = {}
    ignored = 0
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        match = PAIR_STEM.fullmatch(path.stem)
        if not match:
            ignored += 1
            continue
        sample, side = match.group("sample", "side")
        files = samples.setdefault(sample, {})
        if side in files:
            raise ValueError(f"Duplicate files for sample {sample}, side .{side}_1: {files[side]} and {path}")
        files[side] = path

    if ignored:
        print(f"Ignored {ignored} files whose names do not end in .0_1 or .1_1.", file=sys.stderr)
    pairs = {}
    for sample, files in samples.items():
        if len(files) == 2:
            pairs[sample] = (files["0"], files["1"])
        else:
            missing = "1" if "0" in files else "0"
            print(f"Skipping {sample}: missing .{missing}_1.", file=sys.stderr)
    print(f"Complete pairs: {len(pairs)}; incomplete pairs skipped: {len(samples) - len(pairs)}.")
    if not pairs:
        raise ValueError(f"No complete .0_1 / .1_1 pairs found in {directory}.")
    return pairs


def prepare_model(checkpoint_path, device):
    import models_vit

    model = models_vit.vit_large_patch16(
        img_size=224, num_classes=5, drop_path_rate=0, global_pool=True
    )
    checkpoint = load_tensor_file(checkpoint_path)
    model.load_state_dict(checkpoint["model"], strict=False)
    return model.to(device).eval()


def prepare_image(path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB").resize((224, 224))) / 255.0
    mean = pixels.mean(axis=(0, 1), keepdims=True)
    std = pixels.std(axis=(0, 1), keepdims=True)
    pixels = (pixels - mean) / np.maximum(std, 1e-8)
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, epilog="Only use checkpoint and embedding files you trust."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", type=Path, help="Input fundus images.")
    source.add_argument("--embedding-dir", type=Path, help="Existing per-image .pt tensors.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for paired <sample>.pt files.")
    parser.add_argument("--checkpoint", type=Path, help="RETFound checkpoint, required with --input-dir.")
    parser.add_argument("--per-eye-output-dir", type=Path, help="Per-image output; defaults to OUTPUT_DIR/per_eye.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    input_dir = (args.input_dir or args.embedding_dir).resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        parser.error(f"Input directory not found: {input_dir}")
    if input_dir == output_dir:
        parser.error("Choose an output directory distinct from the input directory.")
    if output_dir.is_dir() and any(output_dir.glob("*.pt")):
        parser.error("Choose an output directory without existing .pt files to avoid mixing runs.")

    if args.input_dir:
        if args.checkpoint is None or not args.checkpoint.is_file():
            parser.error("Provide an existing --checkpoint when extracting from images.")
        per_eye_dir = (args.per_eye_output_dir or output_dir / "per_eye").resolve()
        if per_eye_dir in (input_dir, output_dir):
            parser.error("Choose a separate directory for per-image .pt files.")
        if per_eye_dir.is_dir() and any(per_eye_dir.glob("*.pt")):
            parser.error("Choose a per-image output directory without existing .pt files.")
    elif args.checkpoint or args.per_eye_output_dir:
        parser.error("--checkpoint and --per-eye-output-dir only apply with --input-dir.")

    try:
        pairs = find_pairs(input_dir, IMAGE_EXTENSIONS if args.input_dir else {".pt"})
    except ValueError as exc:
        parser.error(str(exc))

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.input_dir:
        from tqdm import tqdm

        per_eye_dir.mkdir(parents=True, exist_ok=True)
        device = torch.device(args.device)
        model = prepare_model(args.checkpoint, device)

        def get_vector(path):
            return eye_vector(model.forward_features(prepare_image(path).to(device)), path)

        items = tqdm(pairs.items(), desc="Extracting paired features")
    else:
        def get_vector(path):
            return eye_vector(load_tensor_file(path), path)

        items = pairs.items()

    with torch.no_grad():
        for sample, (path0, path1) in items:
            vector0, vector1 = get_vector(path0), get_vector(path1)
            if args.input_dir:
                torch.save(vector0, per_eye_dir / f"{path0.stem}.pt")
                torch.save(vector1, per_eye_dir / f"{path1.stem}.pt").
            torch.save(torch.cat((vector0, vector1), dim=0), output_dir / f"{sample}.pt")


if __name__ == "__main__":
    main()
