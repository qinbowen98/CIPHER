#!/usr/bin/env python3
"""Package reviewed code, examples and docs for a protected compute server."""
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    subprocess.run([sys.executable, str(ROOT/'scripts/check_public_release.py')], cwd=ROOT, check=True)
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(set(filter(None, names))):
            path = ROOT/name
            if name.startswith('release/') or path.resolve() == args.output.resolve() or not path.is_file():
                continue
            bundle.write(path, 'CIPHER-code-release/'+name)
    print("Created the code-only bundle.")

if __name__ == '__main__':
    main()
