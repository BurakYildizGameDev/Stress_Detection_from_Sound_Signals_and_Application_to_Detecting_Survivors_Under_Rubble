#!/usr/bin/env python3
"""Manifest builder script."""
import sys
def main():
    print('[*] Building manifest...')
if __name__ == '__main__':
    main()

import argparse
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()
