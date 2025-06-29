# KazKaz Finansal Danışman - Nihai ve Hata Düzeltilmiş Sürüm
import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, auth, firestore
from prophet import Prophet
from prophet.plot import plot_plotly
import google.generativeai as genai
import os

# --- Sayfa Yapılandırması ve Stil (En başta bir kere yapılır) ---
st.set_page_config(page_title="KazKaz Finansal Danışman", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
    body { font-family: 'Segoe UI', sans-serif; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; padding-left: 2rem; padding-right: 2rem; }
    .st-emotion-cache-16txtl3 { background-color: #0f172a; }
    .stButton > button {
        border-radius: 8px; border: 2px solid #10b981; color: white; background-color: #10b981;
        transition: all 0.3s; font-weight: bold; padding: 10px 24px; width: 100%;
    }
    .stButton > button:hover {
        border-color: #34d399; color: white; background-color: #34d399;
    }
    .st-emotion-cache-1gulkj5 { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; }
    h1 { font-size: 3rem; font-weight: 900; color: #ffffff; }
    h2 { font-size: 2.25rem; font-weight: 700; color: #ffffff; }
    h3 { font-size: 1.5rem; font-weight: 600; color: #ffffff; }
    .stTextInput > div > div > input { background-color: #1e293b; color: white; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_firebase():
    try:
        firebase_creds_dict = st.secrets["firebase"]
        cred = credentials.Certificate(firebase_creds_dict)
    except (KeyError, FileNotFoundError):
        try:
            cred = credentials.Certificate("firebase-key.json")
        except FileNotFoundError:
            return None
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    return True

def get_gemini_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key
    try:
        return st.secrets["GEMINI_API_KEY"]
    except:
        return st.sidebar.text_input("Gemini API Anahtarınızı Girin", type="password", help="Bu anahtar sadece yerel testler için gereklidir.")

def calistir_analiz(veri_df):
    if veri_df.empty: return {"hata": "Filtrelenen veri bulunamadı."}
    try:
        toplam_gelir = veri_df['Gelir'].sum()
        toplam_gider = veri_df['Gider'].sum()
        net_kar = toplam_gelir - toplam_gider
        gider_kategorileri = veri_df.groupby('Kategori')['Gider'].sum()
        en_yuksek_gider_kategorisi = gider_kategorileri.idxmax() if not gider_kategorileri.empty else "N/A"
        kar_marji = (net_kar / toplam_gelir * 100) if toplam_gelir > 0 else 0
        aylik_gelir = veri_df.set_index("Tarih")["Gelir"].resample("M").sum()
        ortalama_gelir = aylik_gelir.mean() if not aylik_gelir.empty else 0
        return {
            "toplam_gelir": toplam_gelir,
            "toplam_gider": toplam_gider,
            "net_kar": net_kar,
            "en_yuksek_gider_kategorisi": en_yuksek_gider_kategorisi,
            "kar_marji": kar_marji,
            "ortalama_aylik_gelir": ortalama_gelir
        }
    except Exception as e: return {"hata": str(e)}

def prophet_tahmini_yap(aylik_veri_df):
    if len(aylik_veri_df) < 2: return None, None
    prophet_df = aylik_veri_df.reset_index().rename(columns={'Tarih': 'ds', 'Gelir': 'y'})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=3, freq='M')
    forecast = model.predict(future)
    return model, forecast

def yorum_uret(api_key, analiz_sonuclari, tahmin_trendi):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Sen deneyimli bir finansal danışmansın. Aşağıdaki verilere dayanarak, şirketin durumu hakkında 'içten ve profesyonel' bir tonda, bir durum değerlendirmesi ve 3 maddelik bir eylem planı önerisi yaz.
        Veriler: Toplam Gelir: {analiz_sonuclari['toplam_gelir']:,} TL, Net Kar: {analiz_sonuclari['net_kar']:,} TL, Kar Marjı: %{analiz_sonuclari['kar_marji']:.2f}, Ortalama Aylık Gelir: {analiz_sonuclari['ortalama_aylik_gelir']:,} TL, En Büyük Gider Kalemi: {analiz_sonuclari['en_yuksek_gider_kategorisi']}, Tahmin Trendi: {tahmin_trendi}
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"AI Yorumu üretilemedi: {e}")
        return ""
