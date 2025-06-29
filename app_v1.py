# KazKaz Finansal Danışman - Geliştirilmiş Versiyon

import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, auth, firestore
from prophet import Prophet
from prophet.plot import plot_plotly
import requests
import google.generativeai as genai

# --- Sayfa Yapılandırması ---
st.set_page_config(page_title="KazKaz Finansal Danışman", layout="wide", initial_sidebar_state="expanded")

# --- CSS Stili (Profesyonel, Mobil Uyumluluk ve Renk Paleti Dahil) ---
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}
.st-emotion-cache-1gulkj5 {
    background-color: #1e293b !important;
    border-radius: 16px;
    padding: 20px;
}
.stButton > button {
    border-radius: 8px;
    border: 2px solid #10b981;
    color: #10b981;
    background-color: transparent;
    transition: all 0.3s;
    font-weight: bold;
}
.stButton > button:hover {
    border-color: #ffffff;
    color: #ffffff;
    background-color: #10b981;
}
h1, h2, h3 {
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# --- Firebase Kurulumu ---
try:
    firebase_creds_dict = st.secrets["firebase"]
    cred = credentials.Certificate(firebase_creds_dict)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
except Exception:
    pass

# --- Fonksiyonlar ---
def calistir_analiz(veri_df):
    if veri_df.empty:
        return {"hata": "Veri bulunamadı."}
    try:
        toplam_gelir = veri_df['Gelir'].sum()
        toplam_gider = veri_df['Gider'].sum()
        net_kar = toplam_gelir - toplam_gider
        gider_kategorileri = veri_df.groupby('Kategori')['Gider'].sum()
        en_yuksek_gider_kategorisi = gider_kategorileri.idxmax() if not gider_kategorileri.empty else "N/A"
        return {
            "toplam_gelir": toplam_gelir,
            "toplam_gider": toplam_gider,
            "net_kar": net_kar,
            "en_yuksek_gider_kategorisi": en_yuksek_gider_kategorisi,
            "kar_marji": (net_kar / toplam_gelir * 100) if toplam_gelir else 0,
            "gider_orani": (toplam_gider / toplam_gelir * 100) if toplam_gelir else 0
        }
    except Exception as e:
        return {"hata": str(e)}

def prophet_tahmini_yap(veri):
    if len(veri) < 2:
        return None, None
    prophet_df = veri.reset_index().rename(columns={"Tarih": "ds", "Gelir": "y"})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=3, freq='M')
    forecast = model.predict(future)
    return model, forecast

def yorum_uret(api_key, analiz_sonuclari, tahmin_trendi):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Sen bir şirketin finansal danışmanısın.
    - Gelir: {analiz_sonuclari['toplam_gelir']:,} TL
    - Gider: {analiz_sonuclari['toplam_gider']:,} TL
    - Net Kar: {analiz_sonuclari['net_kar']:,} TL
    - Kar Marjı: %{analiz_sonuclari['kar_marji']:.2f}
    - Gider/Oran: %{analiz_sonuclari['gider_orani']:.2f}
    - En büyük gider: {analiz_sonuclari['en_yuksek_gider_kategorisi']}
    - Gelir tahmini trendi: {tahmin_trendi}
    """
    response = model.generate_content(prompt)
    return response.text

# --- Ana Panel ---
def ana_panel():
    st.title("📊 KazKaz AI Finansal Panel")

    st.markdown("""
    #### 📁 Örnek CSV Şablonu [İndir](https://example.com/ornek.csv)
    Verinizde şu sütunlar olmalıdır: `Tarih`, `Gelir`, `Gider`, `Kategori`
    """)

    uploaded_file = st.file_uploader("CSV dosyanızı yükleyin:", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=['Tarih'])
        analiz = calistir_analiz(df)
        if "hata" not in analiz:
            st.metric("Toplam Gelir", f"{analiz['toplam_gelir']:,} TL")
            st.metric("Toplam Gider", f"{analiz['toplam_gider']:,} TL")
            st.metric("Net Kar", f"{analiz['net_kar']:,} TL")
            st.metric("Kar Marjı", f"%{analiz['kar_marji']:.2f}")
            st.metric("Gider/Oran", f"%{analiz['gider_orani']:.2f}")

            aylik_veri = df.set_index("Tarih")[['Gelir']].resample('ME').sum()
            model, tahmin = prophet_tahmini_yap(aylik_veri)
            if model:
                st.plotly_chart(plot_plotly(model, tahmin))
                trend = "Yükselişte" if tahmin['yhat'].iloc[-1] > tahmin['yhat'].iloc[-4] else "Düşüşte"
                yorum = yorum_uret(st.secrets["GEMINI_API_KEY"], analiz, trend)
                st.subheader("🧠 AI Finansal Yorum")
                st.write(yorum)
            else:
                st.warning("Yeterli veri yok")
        else:
            st.error(analiz["hata"])
    else:
        st.info("Lütfen CSV dosyası yükleyin")

# --- Ana Fonksiyon ---
if __name__ == '__main__':
    ana_panel()
