import os
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from events import (LOG_DIR, STOP_FILE, archive_events, listener_state,
                    read_events, read_status)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LISTENER = os.path.join(BASE_DIR, "live_listener.py")
LISTENER_OUT = os.path.join(LOG_DIR, "listener.out")

STATE_LABELS = {
    "running": "🟢 Çalışıyor",
    "unresponsive": "🟠 Yanıt vermiyor",
    "stopped": "⚪ Kapalı",
    "error": "🔴 Hata",
}
STATE_ICONS = {
    "panic": "🔴",
    "scream": "🔴",
    "stress": "🟠",
    "moan": "🔵",
    "whisper": "🟣",
    "normal": "🟢",
}


def start_listener():
    os.makedirs(LOG_DIR, exist_ok=True)
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    out = open(LISTENER_OUT, "a", encoding="utf-8")
    subprocess.Popen([sys.executable, LISTENER], cwd=BASE_DIR, stdout=out,
                     stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                     **kwargs)


def stop_listener():
    os.makedirs(LOG_DIR, exist_ok=True)
    open(STOP_FILE, "w").close()


def ago(ts):
    sec = (datetime.now() - datetime.fromisoformat(ts)).total_seconds()
    return f"{sec:.0f} sn önce" if sec < 120 else f"{sec / 60:.0f} dk önce"


st.set_page_config(page_title="🚨 Acil Durum Sistemi", layout="wide")

st.title("🚨 Ses Tabanlı Acil Durum Algılama Sistemi")
st.caption("Operatöre dikkat çekmek için bir yardımcıdır; kurtarma kararı vermez. "
           "**Tespit penceresi** tek bir 1 sn'lik pencerenin acil durum sınıfına düşmesidir. "
           "**Alarm**, aynı bölümde art arda yeterli tespit biriktiğinde bir kez üretilir.")

# ======================
# SIDEBAR: DİNLEYİCİ
# ======================
status = read_status()
state = listener_state(status)

st.sidebar.header("🎤 Dinleyici")
st.sidebar.markdown(f"**Durum:** {STATE_LABELS[state]}")
if status:
    st.sidebar.caption(f"Mikrofon: {status.get('device') or 'bilinmiyor'}")
    st.sidebar.caption(f"Son sinyal: {ago(status['last_seen'])}")
    if state == "running":
        level = min(1.0, status.get("last_rms", 0) / 0.2)
        st.sidebar.progress(level, text=f"Giriş seviyesi (RMS {status.get('last_rms', 0):.3f})")
        st.sidebar.caption(f"Son pencere: {status.get('last_status') or '-'} · "
                           f"{status.get('windows_processed', 0)} pencere işlendi")
    if status.get("error"):
        st.sidebar.error(status["error"])
if state == "unresponsive":
    st.sidebar.warning("Dinleyici birkaç saniyedir sinyal göndermiyor. "
                       "Süreç takılmış ya da kapanmış olabilir.")

col_a, col_b = st.sidebar.columns(2)
if col_a.button("▶️ Başlat", disabled=state == "running", width="stretch"):
    start_listener()
    time.sleep(2)
    st.rerun()
if col_b.button("⏹️ Durdur", disabled=state in ("stopped", "error"), width="stretch"):
    stop_listener()
    time.sleep(2)
    st.rerun()
st.sidebar.caption(f"Dinleyici çıktısı: `{os.path.relpath(LISTENER_OUT, BASE_DIR)}`")

st.sidebar.divider()
st.sidebar.header("⚙️ Ayarlar")
refresh_rate = st.sidebar.slider("Yenileme Hızı (saniye)", 1, 10, 2)

if st.sidebar.button("🗄️ Logları Arşivle"):
    try:
        dest = archive_events()
        st.sidebar.success(f"Arşivlendi: {os.path.relpath(dest, BASE_DIR)}" if dest else "Log zaten boş.")
    except PermissionError:
        st.sidebar.warning("Dinleyici o an yazıyordu, tekrar deneyin.")

# ======================
# VERİ
# ======================
events = read_events()
detections = pd.DataFrame([e for e in events if e["type"] == "detection"],
                          columns=["time", "episode", "window", "state", "state_confidence",
                                   "human_prob", "emergency_prob"])
alarms = pd.DataFrame([e for e in events if e["type"] == "alarm"],
                      columns=["time", "episode", "episode_start", "windows", "states",
                               "peak_emergency_prob"])

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🚨 Alarmlar", "🔎 Tespit Pencereleri", "📈 İstatistikler"])

# ======================
# TAB 1: DASHBOARD
# ======================
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎤 Dinleyici", STATE_LABELS[state])
    col2.metric("🚨 Alarm", len(alarms),
                f"Son: {ago(alarms['time'].iloc[-1])}" if len(alarms) else None, delta_color="off")
    col3.metric("🧩 Bölüm", detections["episode"].nunique())
    col4.metric("🔎 Tespit penceresi", len(detections))

    st.divider()
    st.subheader("🚨 Son Alarmlar")
    if alarms.empty:
        st.info("Henüz alarm yok.")
    for _, row in alarms.tail(10).iloc[::-1].iterrows():
        states = ", ".join(f"{STATE_ICONS.get(s, '⚪')} {s} ×{n}" for s, n in row["states"].items())
        st.write(f"**{row['time'][:19].replace('T', ' ')}** · {row['windows']} pencere · "
                 f"en yüksek acil durum olasılığı {row['peak_emergency_prob']:.0%} · {states}")

# ======================
# TAB 2: ALARMLAR
# ======================
with tab2:
    st.subheader("📋 Operatör Alarmları")
    if alarms.empty:
        st.info("Henüz alarm yok.")
    else:
        view = alarms.copy()
        view["states"] = view["states"].apply(lambda d: ", ".join(f"{k}×{v}" for k, v in d.items()))
        view["peak_emergency_prob"] = view["peak_emergency_prob"].apply(lambda x: f"{x:.1%}")
        st.dataframe(view, width="stretch")
        st.download_button("📥 CSV İndir", alarms.to_csv(index=False),
                           file_name=f"alarms_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv")

# ======================
# TAB 3: TESPİT PENCERELERİ
# ======================
with tab3:
    st.subheader("🔎 Tespit Pencereleri")
    st.caption("Tek başına alarm değildir; aynı bölümdeki pencereler tek bir olaya aittir.")
    if detections.empty:
        st.info("Henüz tespit yok.")
    else:
        view = detections.copy()
        for c in ("state_confidence", "human_prob", "emergency_prob"):
            view[c] = view[c].apply(lambda x: f"{x:.1%}")
        st.dataframe(view, width="stretch")
        st.download_button("📥 CSV İndir", detections.to_csv(index=False),
                           file_name=f"detections_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv")

# ======================
# TAB 4: İSTATİSTİKLER
# ======================
with tab4:
    st.subheader("📈 İstatistik Analizi")
    if detections.empty:
        st.info("Henüz veri yok.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(detections["state"].value_counts())
            st.caption("Tespit pencerelerinde sınıf dağılımı")
        with col2:
            st.bar_chart(detections.groupby("episode").size().value_counts().sort_index())
            st.caption("Bölüm uzunluğu (pencere sayısı) dağılımı")

        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Alarm üreten bölüm", f"{alarms['episode'].nunique()} / {detections['episode'].nunique()}")
        col2.metric("Ort. İnsan Olasılığı", f"{detections['human_prob'].mean():.1%}")
        col3.metric("Ort. Acil Durum Olasılığı", f"{detections['emergency_prob'].mean():.1%}")
        col4.metric("Maks. Acil Durum Olasılığı", f"{detections['emergency_prob'].max():.1%}")

# Auto-refresh
st.sidebar.divider()
if st.sidebar.checkbox("🔄 Auto-Refresh", True):
    time.sleep(refresh_rate)
    st.rerun()
