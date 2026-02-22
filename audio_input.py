import sys
import os

# Windows CP1254 konsol unicode desteği
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pipeline import analyze_file

def main():
    if len(sys.argv) < 2:
        print("=" * 50)
        print("[!] Enkaz Alti Ses Analiz Araci")
        print("Kullanim: python audio_input.py <ses_dosyasi.wav/mp3>")
        print("=" * 50)
        sys.exit(1)

    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"[X] HATA: Ses dosyasi bulunamadi: {audio_path}")
        sys.exit(1)

    print(f"[*] '{audio_path}' analiz ediliyor...")
    try:
        result = analyze_file(audio_path)
    except Exception as e:
        print(f"[X] Analiz hatasi: {e}")
        sys.exit(1)

    print("\n" + "=" * 45)
    print("ANALIZ SONUCLARI")
    print("=" * 45)
    status = result.get("status")
    
    if status == "silence":
        print("[-] Durum: SESSIZLIK (Sinyal esik degerinin altinda)")
    elif status == "no_human":
        human_prob = result.get("human_prob", 0.0)
        print(f"[!] Durum: INSAN SESI BULUNAMADI (Insan Olasiligi: %{human_prob*100:.1f})")
    elif status == "DETECTED":
        state = result.get("state", "Bilinmiyor").upper()
        human_prob = result.get("human_prob", 0.0)
        emergency_prob = result.get("emergency_prob", 0.0)
        is_emergency = result.get("is_emergency", False)

        alert_mark = "[ALARM]" if is_emergency else "[NORMAL]"
        print(f"[+] Canli Varligi: TESPIT EDILDI (Guven: %{human_prob*100:.1f})")
        print(f"[+] Durum: {state}")
        print(f"[+] Acil Durum Olasiligi: %{emergency_prob*100:.1f}")
        print(f"{alert_mark} Acil Durum Alarmi: {'EVET' if is_emergency else 'HAYIR'}")
        
        all_probs = result.get("all_probs", {})
        if all_probs:
            print("\nSinif Dagilimi:")
            for cls, prob in all_probs.items():
                print(f"   - {cls.capitalize():8}: %{prob*100:.1f}")
    print("=" * 45 + "\n")

if __name__ == "__main__":
    main()
