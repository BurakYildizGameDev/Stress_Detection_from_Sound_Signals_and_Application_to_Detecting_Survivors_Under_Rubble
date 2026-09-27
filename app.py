import streamlit as st
import json
import pandas as pd
from datetime import datetime
import time
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "alerts_log.json")

# live_listener.py son alert'ten bu kadar saniye sonra "sessiz" sayılır
ACTIVE_WINDOW_SEC = 30


def load_alerts():
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def seconds_since(timestamp):
    try:
        t = datetime.strptime(timestamp[:19], "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None
    return (datetime.now() - t).total_seconds()


st.set_page_config(page_title="🚨 Acil Durum Sistemi", layout="wide")

st.title("🚨 Ses Tabanlı Acil Durum Algılama Sistemi")
st.caption("Bu panel `live_listener.py` tarafından yazılan `alerts_log.json` dosyasını gösterir. "
           "Dinleyiciyi ayrı bir terminalde başlatın: `python live_listener.py`")

# Sidebar ayarları
st.sidebar.header("⚙️ Ayarlar")
refresh_rate = st.sidebar.slider("Yenileme Hızı (saniye)", 1, 10, 2)

if st.sidebar.button("🗑️ Logları Temizle"):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)
    st.sidebar.success("Loglar temizlendi!")
    time.sleep(1)
    st.rerun()

alerts = load_alerts()
df = pd.DataFrame(alerts, columns=["timestamp", "state", "human_prob", "emergency_prob"])

# Ana sekmeleri oluştur
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🚨 Alert Logları", "📈 İstatistikler"])

# ======================
# TAB 1: DASHBOARD
# ======================
with tab1:
    col1, col2, col3 = st.columns(3)

    last_ts = alerts[-1]["timestamp"] if alerts else None
    age = seconds_since(last_ts) if last_ts else None

    with col1:
        if age is not None and age <= ACTIVE_WINDOW_SEC:
            st.metric("🎤 Son Aktivite", "Yeni alert 🔴", f"{age:.0f} sn önce")
        elif age is not None:
            st.metric("🎤 Son Aktivite", "Sessiz", f"{age / 60:.0f} dk önce", delta_color="off")
        else:
            st.metric("🎤 Son Aktivite", "Kayıt yok")

    with col2:
        st.metric("🚨 Toplam Alert", len(alerts), f"Son: {last_ts or 'N/A'}", delta_color="off")

    with col3:
        st.metric("⏰ Son Güncelleme", datetime.now().strftime("%H:%M:%S"))

    st.divider()

    # İstatistik kartları
    states = df["state"].value_counts().to_dict()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("😨 Panik", states.get('panic', 0))
    with col2:
        st.metric("😱 Çığlık", states.get('scream', 0))
    with col3:
        st.metric("😟 Stres", states.get('stress', 0))
    with col4:
        st.metric("😊 Normal", states.get('normal', 0))

    st.divider()

    st.subheader("📉 Son Alert'ler")

    def color_state(state):
        colors = {
            'panic': '🔴',
            'scream': '🔴',
            'stress': '🟠',
            'normal': '🟢'
        }
        return colors.get(state, '⚪')

    if df.empty:
        st.info("Henüz alert kaydı yok.")
    for _, row in df.tail(10).iterrows():
        col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
        with col1:
            st.write(color_state(row['state']))
        with col2:
            st.write(f"**{str(row['state']).upper()}** - {row['timestamp']}")
        with col3:
            st.write(f"Human: {row['human_prob']*100:.1f}%")
        with col4:
            st.write(f"Emergency: {row['emergency_prob']*100:.1f}%")

# ======================
# TAB 2: ALERT LOGLAR
# ======================
with tab2:
    st.subheader("📋 Tüm Alert Logları")

    if df.empty:
        st.info("Henüz alert kaydı yok.")
    else:
        view = df.copy()
        view['human_prob'] = view['human_prob'].apply(lambda x: f"{x*100:.1f}%")
        view['emergency_prob'] = view['emergency_prob'].apply(lambda x: f"{x*100:.1f}%")

        st.dataframe(view, width="stretch")

        st.download_button(
            label="📥 CSV İndir",
            data=df.to_csv(index=False),
            file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# ======================
# TAB 3: İSTATİSTİKLER
# ======================
with tab3:
    st.subheader("📈 İstatistik Analizi")

    if df.empty:
        st.info("Henüz veri yok.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.bar_chart(df['state'].value_counts())
            st.caption("State Dağılımı")

        with col2:
            avg_probs = {
                'Human Prob Ort.': df['human_prob'].mean() * 100,
                'Emergency Prob Ort.': df['emergency_prob'].mean() * 100
            }
            st.bar_chart(pd.Series(avg_probs))
            st.caption("Ortalama Olasılıklar")

        st.divider()
        st.subheader("📊 Özet")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Toplam Alert", len(df))
        with col2:
            st.metric("Ort. Human Prob", f"{df['human_prob'].mean()*100:.1f}%")
        with col3:
            st.metric("Ort. Emergency Prob", f"{df['emergency_prob'].mean()*100:.1f}%")
        with col4:
            st.metric("Max Emergency Prob", f"{df['emergency_prob'].max()*100:.1f}%")

# Auto-refresh
st.sidebar.divider()
if st.sidebar.checkbox("🔄 Auto-Refresh", True):
    time.sleep(refresh_rate)
    st.rerun()
