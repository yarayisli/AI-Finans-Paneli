import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, auth, firestore
from prophet import Prophet
from prophet.plot import plot_plotly
import requests
import google.generativeai as genai

# --- Sayfa Yapılandırması ve Stil ---
st.set_page_config(page_title="KazKaz Finansal Danışman", layout="wide", initial_sidebar_state="expanded")

# --- CSS Stilleri (Değişiklik yok) ---
st.markdown("""
<style>
    /* ... (Önceki stillerimiz burada) ... */
    .stButton > button { border-radius: 8px; border: 2px solid #10b981; color: #10b981; background-color: transparent; transition: all 0.3s; font-weight: bold; }
    .stButton > button:hover { border-color: #ffffff; color: #ffffff; background-color: #10b981; }
    .st-emotion-cache-1gulkj5 { background-color: #1e293b; border-radius: 12px; padding: 20px; }
    h1, h2, h3 { color: #ffffff; }
</style>
""", unsafe_allow_html=True)


# --- Firebase Bağlantısı (Hibrit) ---
try:
    # Önce Streamlit Cloud Secrets'ı dene
    firebase_creds_dict = st.secrets["firebase"]
    cred = credentials.Certificate(firebase_creds_dict)
except (KeyError, FileNotFoundError):
    # Eğer bulamazsa, yereldeki anahtar dosyasını dene
    try:
        cred = credentials.Certificate("firebase-key.json")
    except FileNotFoundError:
        st.warning("Firebase anahtarı bulunamadı. Lütfen `firebase-key.json` dosyasını ana dizine ekleyin veya Streamlit Secrets'ı yapılandırın.")
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

def yorum_uret(api_key, analiz_sonuclari, tahmin_ozeti):
    try:
        genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Sen deneyimli bir finansal danışmansın. Aşağıdaki verilere dayanarak, şirketin durumu hakkında 'içten, sıcak ve insani' bir tonda, bir durum değerlendirmesi ve eylem planı önerisi yaz.
        Veriler: Toplam Gelir: {analiz_sonuclari['toplam_gelir']:,} TL, Net Kar: {analiz_sonuclari['net_kar']:,} TL, En Büyük Gider Kalemi: {analiz_sonuclari['en_yuksek_gider_kategorisi']}, Tahmin Trendi: {tahmin_ozeti}
        Yorumuna "Değerli Yönetici," diye başla.
        """
        response = model.generate_content(prompt); return response.text
    except Exception as e:
        st.error(f"AI Yorumu üretilemedi: {e}"); return ""

# --- ARAYÜZ FONKSİYONLARI ---
def show_landing_page():
    # ... (Ana sayfa fonksiyonu aynı)
    st.title("Finansal Özgürlüğe Ulaşmanın En Kolay Yolu")
    st.subheader("KazKaz Finansal Danışman ile verilerinizi anlamlı bilgilere ve stratejilere dönüştürün.")
    if st.button("Hemen Başla", type="primary"):
        st.session_state['page'] = 'login'
        st.rerun()

def show_dashboard(subscription_plan, api_key): # DİKKAT: Artık api_key'i parametre olarak alıyor
    st.title(f"🚀 Finansal Analiz Paneli ({subscription_plan} Paket)")
    uploaded_file = st.file_uploader("Analiz için CSV dosyanızı buraya yükleyin", type="csv")
    if uploaded_file:
        ana_veri = pd.read_csv(uploaded_file, parse_dates=['Tarih'])
        analiz_sonuclari = calistir_analiz(ana_veri)
        if "hata" not in analiz_sonuclari:
            st.header("Genel Finansal Durum")
            col1, col2, col3 = st.columns(3)
            col1.metric("Toplam Gelir", f"{analiz_sonuclari['toplam_gelir']:,} TL")
            col2.metric("Toplam Gider", f"{analiz_sonuclari['toplam_gider']:,} TL")
            col3.metric("Net Kar", f"{analiz_sonuclari['net_kar']:,} TL")
            
            st.divider()
            if subscription_plan in ['Pro', 'Enterprise']:
                st.header("🤖 AI Danışman Yorumu")
                with st.spinner("Yapay zeka danışmanınız verileri analiz ediyor..."):
                    aylik_veri = ana_veri.set_index('Tarih')[['Gelir']].resample('ME').sum()
                    model, tahmin = prophet_tahmini_yap(aylik_veri)
                    if tahmin is not None:
                        tahmin_trendi = "Yükselişte" if tahmin['yhat'].iloc[-1] > tahmin['yhat'].iloc[-4] else "Düşüşte veya Durgun"
                        # DİKKAT: Fonksiyona parametre olarak gelen api_key kullanılıyor
                        yorum = yorum_uret(api_key, analiz_sonuclari, tahmin_trendi)
                        st.markdown(yorum)
                    else:
                        st.warning("Yorum için yeterli veri yok.")
            else:
                st.info("AI Danışman Yorumu 'Pro' ve 'Enterprise' paketlerinde mevcuttur.")

def main():
    if 'user_info' not in st.session_state: st.session_state['user_info'] = None
    if 'page' not in st.session_state: st.session_state['page'] = 'landing'
    
    # YENİ: Gemini API anahtarını hibrit yöntemle alma
    gemini_api_key = None
    try:
        # Önce Cloud Secrets'ı dene
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        # Bulamazsa kullanıcıdan iste (sadece yerelde çalışır)
        st.sidebar.subheader("Yerel Geliştirme Ayarı")
        gemini_api_key = st.sidebar.text_input("Gemini API Anahtarınızı Girin", type="password", help="Bu anahtar sadece yerel testler için gereklidir.")


    if st.session_state['user_info']:
        # KULLANICI GİRİŞ YAPMIŞSA
        user_uid = st.session_state['user_info']['uid']
        st.sidebar.subheader(f"Hoş Geldin, {st.session_state['user_info']['email']}")
        if st.sidebar.button("Çıkış Yap"):
            st.session_state.clear(); st.rerun()
        
        db = firestore.client()
        user_doc_ref = db.collection('users').document(user_uid)
        user_doc = user_doc_ref.get()
        subscription_plan = user_doc.to_dict().get('subscription_plan', 'None')
        
        if subscription_plan == 'None':
            # Fiyatlandırma ekranı...
            st.title("Abonelik Paketleri")
            if st.button("Pro Paket Seç (₺750/ay)"): user_doc_ref.set({'subscription_plan': 'Pro'}, merge=True); st.rerun()
        else:
            # DİKKAT: Dashboard fonksiyonuna api_key'i de gönderiyoruz
            show_dashboard(subscription_plan, gemini_api_key)
    else:
        # GİRİŞ YAPMAMIŞ KULLANICI
        if st.session_state['page'] == 'landing':
            show_landing_page()
        else:
            # Giriş/Kayıt ekranı...
            st.subheader("Giriş Yap veya Kayıt Ol")
            # ... (Giriş/Kayıt mantığı aynı)
            db = firestore.client()
            email = st.text_input("E-posta")
            password = st.text_input("Şifre", type="password")
            if st.button("Giriş Yap", type="primary"):
                try:
                    user = auth.get_user_by_email(email)
                    st.session_state['user_info'] = {'uid': user.uid, 'email': user.email}; st.rerun()
                except: st.error("E-posta veya şifre hatalı.")

if __name__ == '__main__':
    main()
