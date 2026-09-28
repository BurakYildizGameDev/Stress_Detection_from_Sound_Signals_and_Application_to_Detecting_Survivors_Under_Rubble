"""
c_compiler.py - Bilgisayardaki C derleyicisini bulur (testler ve esp_simulate.py için).

Sıra: CC ortam değişkeni (ör. "gcc" ya da "python -m ziglang cc"),
PATH'teki gcc / cc / clang, son olarak kuruluysa ziglang paketi
(Windows'ta en kolay yol: pip install ziglang).
"""
import importlib.util
import os
import shlex
import shutil
import sys


def find_c_compiler():
    """Derleyici komutunu liste olarak döndürür; bulunamazsa None."""
    if os.environ.get("CC"):
        return shlex.split(os.environ["CC"], posix=os.name != "nt")
    for name in ("gcc", "cc", "clang"):
        path = shutil.which(name)
        if path:
            return [path]
    if importlib.util.find_spec("ziglang") is not None:
        return [sys.executable, "-m", "ziglang", "cc"]
    return None
