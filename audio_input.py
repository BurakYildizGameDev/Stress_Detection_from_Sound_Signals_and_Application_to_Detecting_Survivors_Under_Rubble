import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def main():
    print('[!] Acoustic Survivor Detection Input Tool')

if __name__ == '__main__':
    main()
