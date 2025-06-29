# KazKaz AI Finansal Danışman - Profesyonel Site Versiyonu (Hibrit Düzeltme)
import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, auth, firestore
from prophet import Prophet
from prophet.plot import plot_plotly
import requests
import google.generativeai as genai

# --- Sayfa Yapılandırması ve Stil ---
st.set_page_config(page_title="KazKaz Finansal Danışman", layout="wide", initial_sidebar_state="collapsed")

# --- CSS Stilleri ---
st.markdown("""
<style>
    /* ... (CSS Stillerimiz aynı kalıyor) ... */
    .stButton > button { border-radius: 8px; border: 2px solid #10b981; color: white; background-color: #10b981; transition: all 0.3s; font-weight: bold; padding: 10px 24px; }
    .stButton > button:hover { border-color: #34d399; color: white; background-color: #34d399; }
    .st-emotion-cache-1gulkj5 { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; }
    h1 { font-size: 3rem; font-weight: 900; }
    h2 { font-size: 2.25rem; font-weight: 700; }
    h3 { font-size: 1.5rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# --- Firebase Bağlantısı (Hibrit: Yerel + Bulut) ---
try:
    # Önce Streamlit Cloud Secrets'ı dene
    firebase_creds_dict = st.secrets["firebase"]
    cred = credentials.Certificate(firebase_creds_dict)
except (KeyError, FileNotFoundError):
    # Eğer bulamazsa, yereldeki anahtar dosyasını dene
    try:
        cred = credentials.Certificate("firebase-key.json")
    except FileNotFoundError:
        # Anahtar bulunamazsa devam et, hata verme
        cred = None

if cred and not firebase_admin._apps:
    firebase_admin.initialize_app(cred)


# --- Tüm Analiz Fonksiyonları (Değişiklik yok) ---
def calistir_analiz(veri_df):
    if veri_df.empty: return {"hata": "Filtrelenen veri bulunamadı."}
    try:
        toplam_gelir = veri_df['Gelir'].sum(); toplam_gider = veri_df['Gider'].sum(); net_kar = toplam_gelir - toplam_gider
        gider_kategorileri = veri_df.groupby('Kategori')['Gider'].sum()
        en_yuksek_gider_kategorisi = gider_kategorileri.idxmax() if not gider_kategorileri.empty else "N/A"
        return {"toplam_gelir": toplam_gelir, "toplam_gider": toplam_gider, "net_kar": net_kar, "en_yuksek_gider_kategorisi": en_yuksek_gider_kategorisi}
    except Exception as e: return {"hata": str(e)}

def prophet_tahmini_yap(aylik_veri_df):
    if len(aylik_veri_df) < 2: return None, None
    prophet_df = aylik_veri_df.reset_index().rename(columns={'Tarih': 'ds', 'Gelir': 'y'})
    model = Prophet(); model.fit(prophet_df)
    future = model.make_future_dataframe(periods=3, freq='M')
    forecast = model.predict(future)
    return model, forecast

def yorum_uret(api_key, analiz_sonuclari, tahmin_trendi):
    try:
        genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Sen deneyimli bir finansal danışmansın. Aşağıdaki verilere dayanarak, şirketin durumu hakkında 'içten ve profesyonel' bir tonda, bir durum değerlendirmesi ve 3 maddelik bir eylem planı önerisi yaz.
        Veriler: Toplam Gelir: {analiz_sonuclari['toplam_gelir']:,} TL, Net Kar: {analiz_sonuclari['net_kar']:,} TL, En Büyük Gider Kalemi: {analiz_sonuclari['en_yuksek_gider_kategorisi']}, Tahmin Trendi: {tahmin_trendi}
        """
        response = model.generate_content(prompt); return response.text
    except Exception as e:
        st.error(f"AI Yorumu üretilemedi: {e}"); return ""


# --- ARAYÜZ FONKSİYONLARI ---
def show_landing_page():
    st.title("Finansal Verilerinizi **Anlamlı Stratejilere** Dönüştürün")
    st.subheader("KazKaz AI, işletmenizin finansal sağlığını analiz eder, geleceği tahminler ve size özel eylem planları sunar.")
    st.write(" ")
    if st.button("🚀 Ücretsiz Denemeye Başla"):
        st.session_state['page'] = 'login'
        st.rerun()

def show_dashboard(subscription_plan, api_key):
    st.sidebar.title("Kontrol Paneli")
    st.title(f"🚀 Finansal Analiz Paneli")
    st.sidebar.info(f"Aktif Paketiniz: **{subscription_plan}**")
    
    uploaded_file = st.sidebar.file_uploader("CSV dosyanızı yükleyin", type="csv")
    if uploaded_file:
        # Analiz paneli kodları
        # ...
        if subscription_plan in ['Pro', 'Enterprise']:
            if api_key: # Sadece anahtar varsa AI yorumunu göster
                # ... Yorum üretme ve gösterme ...
                st.header("🤖 AI Danışman Yorumu")
                # ...
            else:
                st.sidebar.warning("AI Yorumu için Gemini API anahtarınızı girmeniz gerekmektedir.")


def main():
    if 'user_info' not in st.session_state: st.session_state['user_info'] = None
    if 'page' not in st.session_state: st.session_state['page'] = 'landing'

    if st.session_state['user_info']:
        # KULLANICI GİRİŞ YAPMIŞSA
        user_uid = st.session_state['user_info']['uid']
        db = firestore.client()
        user_doc_ref = db.collection('users').document(user_uid)
        user_doc = user_doc_ref.get()
        subscription_plan = user_doc.to_dict().get('subscription_plan', 'None')
        
        # Kenar çubuğu
        st.sidebar.subheader(f"Hoş Geldin, {st.session_state['user_info']['email']}")
        if st.sidebar.button("Çıkış Yap"):
            st.session_state.clear(); st.rerun()

        # DÜZELTİLMİŞ: Hibrit Gemini API Anahtar Yönetimi
        gemini_api_key = None
        try:
            gemini_api_key = st.secrets["GEMINI_API_KEY"]
        except (KeyError, FileNotFoundError):
            st.sidebar.warning("Cloud anahtarı bulunamadı.")
            gemini_api_key = st.sidebar.text_input("Gemini API Anahtarınızı Girin", type="password", help="Bu sadece yerel testler içindir.")

        if subscription_plan == 'None':
            # Fiyatlandırma ekranı...
            st.title("Abonelik Paketleri")
            # ...
        else:
            show_dashboard(subscription_plan, gemini_api_key)
    else:
        # GİRİŞ YAPMAMIŞ KULLANICI
        if st.session_state['page'] == 'landing':
            show_landing_page()
        else:
            # Giriş/Kayıt ekranı...
            st.subheader("Hesabınıza Giriş Yapın")
            # ...

if __name__ == '__main__':
    main()
