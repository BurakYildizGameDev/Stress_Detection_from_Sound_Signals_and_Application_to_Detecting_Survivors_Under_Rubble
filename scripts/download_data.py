#!/usr/bin/env python3
"""Download datasets for acoustic survivor detection."""
import os, sys

def main():
    print('[*] Dataset download script initialized.')

if __name__ == '__main__':
    main()

DATASETS = {
    'esc50': {'name': 'ESC-50', 'url': 'https://github.com/karolpiczak/ESC-50/archive/master.zip'},
    'ravdess': {'name': 'RAVDESS', 'url': 'https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip'},
}

def download_file(url, target_path):
    print(f'Downloading {url} to {target_path}...')

import zipfile
def extract_archive(archive_path, extract_dir):
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, 'r') as z:
            z.extractall(extract_dir)
