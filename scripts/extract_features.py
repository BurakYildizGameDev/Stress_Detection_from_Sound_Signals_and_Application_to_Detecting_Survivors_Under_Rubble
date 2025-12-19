#!/usr/bin/env python3
"""Offline feature extraction script."""
import sys
def main():
    print('[*] Extracting features...')
if __name__ == '__main__':
    main()

from multiprocessing import Pool, cpu_count
def run_parallel(tasks, n_workers=None):
    n = n_workers or cpu_count()
    print(f'Using {n} workers.')
