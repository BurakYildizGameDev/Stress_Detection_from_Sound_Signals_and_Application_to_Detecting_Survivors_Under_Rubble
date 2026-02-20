import sys, os
from pipeline import analyze_file
def main():
    if len(sys.argv) < 2:
        print('Kullanim: python audio_input.py <ses>')
        sys.exit(1)
    res = analyze_file(sys.argv[1])
    print(res)
if __name__ == '__main__':
    main()
