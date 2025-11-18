import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def main():
    if len(sys.argv) < 2:
        print('Kullanim: python audio_input.py <ses_dosyasi.wav/mp3>')
        sys.exit(1)
    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f'Hata: {audio_path} bulunamadi')
        sys.exit(1)
    print(f'[*] Analiz ediliyor: {audio_path}')

if __name__ == '__main__':
    main()
