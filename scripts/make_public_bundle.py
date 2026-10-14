#!/usr/bin/env python3
"""Bundle English source plus the two receipted model weights; exclude private files."""
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.check_public_release import release_files

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    subprocess.run([sys.executable,str(ROOT/'scripts/check_public_release.py')],cwd=ROOT,check=True)
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    names=set(filter(None,names))|release_files(ROOT/'release')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            path=ROOT/name
            if path.is_file() and path.resolve()!=args.output.resolve():
                archive.write(path,'CIPHER/'+name)
    print('Created the source and final-model bundle.')


if __name__=='__main__':main()
