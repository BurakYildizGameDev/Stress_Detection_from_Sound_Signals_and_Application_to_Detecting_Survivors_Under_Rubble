#!/usr/bin/env python3
"""Model training script."""
def main():
    print('[*] Training models...')
if __name__ == '__main__':
    main()

import argparse
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n-trees', type=int, default=300)
    return p.parse_args()
