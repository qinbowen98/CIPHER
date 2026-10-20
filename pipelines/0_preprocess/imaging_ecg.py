"""Prepare square fundus images and fixed-length ECG XML files."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


def center_square(image):
    width, height = image.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def normalize_lead(values, method):
    values = np.asarray(values, dtype=np.float32)
    if not values.size or not np.isfinite(values).all():
        raise ValueError('ECG leads must contain finite samples.')
    if method == 'zscore':
        scale = float(values.std())
        return (values - values.mean()) / scale if scale > 0 else np.zeros_like(values)
    if method == 'maxabs':
        scale = float(np.abs(values).max())
        return values / scale if scale > 0 else np.zeros_like(values)
    if method == 'none':
        return values
    raise ValueError('Unknown ECG normalization.')


def prepare_ecg(leads, length=1000, normalization='zscore'):
    if length < 1 or len(leads) != 12:
        raise ValueError('A positive length and exactly 12 ordered ECG leads are required.')
    output = np.zeros((length, 12), dtype=np.float32)
    for column, lead in enumerate(leads):
        valid = np.asarray(lead, dtype=np.float32)[:length]
        normalized = normalize_lead(valid, normalization)
        output[:len(normalized), column] = normalized
    return output


def read_ecg(path, length=1000, normalization='none'):
    tree = ET.parse(path)
    nodes = [seq.find('value/digits') for seq in tree.getroot().findall('.//sequence')]
    leads = [list(map(float, node.text.split())) for node in nodes if node is not None and node.text]
    return prepare_ecg(leads, length, normalization)


def fundus_transform(image_size=224, normalization='imagenet'):
    from torchvision import transforms
    steps = [transforms.Lambda(center_square), transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    if normalization == 'imagenet':
        steps.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    elif normalization == 'half':
        steps.append(transforms.Normalize([0.5] * 3, [0.5] * 3))
    elif normalization != 'none':
        raise ValueError('Unknown image normalization.')
    return transforms.Compose(steps)


def run(args):
    source, output = Path(args.input_dir).resolve(), Path(args.output_dir).resolve()
    if not source.is_dir() or source == output or source in output.parents:
        raise ValueError('Use an existing input directory and a separate output directory.')
    paths = sorted(p for p in source.iterdir() if p.suffix.lower() in
                   ({'.xml'} if args.kind == 'ecg' else {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}))
    if not paths:
        raise ValueError('No supported inputs found.')
    output.mkdir(parents=True, exist_ok=False)
    for path in paths:
        if args.kind == 'fundus':
            with Image.open(path) as image:
                image = center_square(image.convert('RGB')).resize((args.image_size, args.image_size), Image.Resampling.BILINEAR)
                image.save(output / (path.stem + '.png'))
        else:
            tree = ET.parse(path)
            nodes = [seq.find('value/digits') for seq in tree.getroot().findall('.//sequence')]
            nodes = [node for node in nodes if node is not None and node.text]
            if len(nodes) != 12:
                raise ValueError('Each ECG must contain exactly 12 ordered digit sequences.')
            leads = [list(map(float, node.text.split())) for node in nodes]
            prepared = prepare_ecg(leads, args.length, args.normalization)
            for column, node in enumerate(nodes):
                node.text = ' '.join(format(float(value), '.9g') for value in prepared[:, column])
            tree.write(output / path.name, encoding='utf-8', xml_declaration=True)
    manifest = {'kind': args.kind, 'n_files': len(paths), 'image_size': args.image_size,
                'crop': 'center square', 'ecg_length': args.length,
                'ecg_normalization': args.normalization, 'normalization_scope': 'each retained lead before padding',
                'image_normalization': 'applied by the configured training transform',
                'participant_outputs': True}
    (output / 'preprocessing.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Prepared files:', len(paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['fundus', 'ecg'], required=True)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--length', type=int, default=1000)
    parser.add_argument('--normalization', choices=['none', 'zscore', 'maxabs'], default='zscore')
    args = parser.parse_args()
    if args.image_size < 1:
        parser.error('image-size must be positive.')
    run(args)


if __name__ == '__main__':
    main()
