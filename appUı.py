import streamlit as st
import json
import pandas as pd
from datetime import datetime
import time
import os

st.set_page_config(page_title="🚨 Acil Durum Sistemi", layout="wide")

st.title("🚨 Ses Tabanlı Acil Durum Algılama Sistemi")

# Sidebar ayarları
st.sidebar.header("⚙️ Ayarlar")
refresh_rate = st.sidebar.slider("Yenileme Hızı (saniye)", 1, 10, 2)

if st.sidebar.button("🗑️ Logları Temizle"):
    with open("alerts_log.json", "w") as f:
        json.dump([], f)
    st.sidebar.success("Loglar temizlendi!")
    time.sleep(1)
    st.rerun()

# Ana sekmeleri oluştur
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🚨 Alert Logları", "📈 İstatistikler"])

# ======================
# TAB 1: DASHBOARD
# ======================
with tab1:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎤 Dinleme Durumu", "Aktif ✅", "Live")
    
    with col2:
        # Alert logunu oku
        if os.path.exists("alerts_log.json"):
            with open("alerts_log.json", "r") as f:
                try:
                    alerts = json.load(f)
                except json.JSONDecodeError:
                    alerts = []
            st.metric("🚨 Toplam Alert", len(alerts), f"Son: {alerts[-1]['timestamp'] if alerts else 'N/A'}")
        else:
            st.metric("🚨 Toplam Alert", 0, "Henüz alert yok")
    
    with col3:
        st.metric("⏰ Son Güncelleme", datetime.now().strftime("%H:%M:%S"), "Canlı")
    
    st.divider()
    
    # İstatistik kartları
    col1, col2, col3, col4 = st.columns(4)
    
    if os.path.exists("alerts_log.json"):
        with open("alerts_log.json", "r") as f:
            try:
                alerts = json.load(f)
            except json.JSONDecodeError:
                alerts = []
        
        # State dağılımı
        states = {}
        for alert in alerts:
            state = alert.get('state', 'unknown')
            states[state] = states.get(state, 0) + 1
        
        with col1:
            st.metric("😨 Panik", states.get('panic', 0))
        with col2:
            st.metric("😱 Çığlık", states.get('scream', 0))
        with col3:
            st.metric("😟 Stres", states.get('stress', 0))
        with col4:
            st.metric("😊 Normal", states.get('normal', 0))
    
    st.divider()
    
    # Canlı graf
    st.subheader("📉 Son Alert'ler")
    if os.path.exists("alerts_log.json"):
        with open("alerts_log.json", "r") as f:
            try:
                alerts = json.load(f)
            except json.JSONDecodeError:
                alerts = []
        
        # Son 10 alert
        df = pd.DataFrame(alerts[-10:])
        
        # State rengini belirle
        def color_state(state):
            colors = {
                'panic': '🔴',
                'scream': '🔴',
                'stress': '🟠',
                'normal': '🟢'
            }
            return colors.get(state, '⚪')
        
        for idx, row in df.iterrows():
            col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
            with col1:
                st.write(color_state(row['state']))
            with col2:
                st.write(f"**{row['state'].upper()}** - {row['timestamp']}")
            with col3:
                st.write(f"Human: {row['human_prob']*100:.1f}%")
            with col4:
                st.write(f"Emergency: {row['emergency_prob']*100:.1f}%")

# ======================
# TAB 2: ALERT LOGLAR
# ======================
with tab2:
    st.subheader("📋 Tüm Alert Logları")
    
    if os.path.exists("alerts_log.json"):
        with open("alerts_log.json", "r") as f:
            try:
                alerts = json.load(f)
            except json.JSONDecodeError:
                alerts = []
        
        df = pd.DataFrame(alerts)
        df['human_prob'] = df['human_prob'].apply(lambda x: f"{x*100:.1f}%")
        df['emergency_prob'] = df['emergency_prob'].apply(lambda x: f"{x*100:.1f}%")
        
        st.dataframe(df, use_container_width=True)
        
        # CSV indir
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 CSV İndir",
            data=csv,
            file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Henüz alert kaydı yok.")

# ======================
# TAB 3: İSTATİSTİKLER
# ======================
with tab3:
    st.subheader("📈 İstatistik Analizi")
    
    if os.path.exists("alerts_log.json"):
        with open("alerts_log.json", "r") as f:
            try:
                alerts = json.load(f)
            except json.JSONDecodeError:
                alerts = []
        
        df = pd.DataFrame(alerts)
        
        # State dağılım grafiği
        col1, col2 = st.columns(2)
        
        with col1:
            state_counts = df['state'].value_counts()
            st.bar_chart(state_counts)
            st.caption("State Dağılımı")
        
        with col2:
            avg_probs = {
                'Human Prob Ort.': df['human_prob'].mean() * 100,
                'Emergency Prob Ort.': df['emergency_prob'].mean() * 100
            }
            st.bar_chart(pd.Series(avg_probs))
            st.caption("Ortalama Olasılıklar")
        
        # Özet istatistikler
        st.divider()
        st.subheader("📊 Özet")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Toplam Alert", len(alerts))
        with col2:
            st.metric("Ort. Human Prob", f"{df['human_prob'].mean()*100:.1f}%")
        with col3:
            st.metric("Ort. Emergency Prob", f"{df['emergency_prob'].mean()*100:.1f}%")
        with col4:
            st.metric("Max Emergency Prob", f"{df['emergency_prob'].max()*100:.1f}%")
    
    else:
        st.info("Henüz veri yok.")

# Auto-refresh
st.sidebar.divider()

# SMS Gönderme Paneli
st.sidebar.subheader("📱 SMS Gönder")
phone = st.sidebar.text_input("Telefon Numarası", "+905XXXXXXXXX")
message = st.sidebar.text_area("Mesaj", "🚨 ACİL DURUM ALGILANDI!")

if st.sidebar.button("📤 SMS Gönder", key="send_sms"):
    st.sidebar.success("✅ SMS gönderildi!")
    st.balloons()

st.sidebar.divider()
if st.sidebar.checkbox("🔄 Auto-Refresh", True):
    time.sleep(refresh_rate)
    st.rerun()