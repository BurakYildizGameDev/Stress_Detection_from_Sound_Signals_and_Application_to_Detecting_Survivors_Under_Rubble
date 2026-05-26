import sys
from pipeline import analyze_audio

if len(sys.argv) < 2:
    print("Kullanım: python audio_input.py <ses.wav>")
    sys.exit(1)

result = analyze_audio(sys.argv[1])

print("\nSONUÇ:")
for k, v in result.items():
    print(f"{k}: {v}")
