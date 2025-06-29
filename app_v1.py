# KazKaz AI Finansal Danışman - Nihai ve Tam Sürüm
import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, auth, firestore
from prophet import Prophet
from prophet.plot import plot_plotly
import google.generativeai as genai

# --- Sayfa Yapılandırması ve Stil ---
st.set_page_config(page_title="KazKaz Finansal Danışman", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
    /* ... (CSS Stillerimiz aynı kalıyor) ... */
    .stButton > button { border-radius: 8px; border: 2px solid #10b981; color: white; background-color: #10b981; transition: all 0.3s; font-weight: bold; padding: 10px 24px; }
    .stButton > button:hover { border-color: #34d399; color: white; background-color: #34d399; }
    h1 { font-size: 3rem; font-weight: 900; }
    .st-emotion-cache-1gulkj5 { background-color: #1e293b; border-radius: 12px; padding: 20px; }
</style>
""", unsafe_allow_html=True)


# --- GÜVENLİ BAĞLANTI VE ANAHTAR YÖNETİMİ ---

@st.cache_resource
def init_firebase():
    """
    Firebase bağlantısını güvenli bir şekilde başlatır.
    Hem yerelde (dosyadan) hem de bulutta (secrets'tan) çalışır.
    """
    try:
        firebase_creds_dict = st.secrets["firebase"]
        cred = credentials.Certificate(firebase_creds_dict)
    except (KeyError, FileNotFoundError):
        try:
            cred = credentials.Certificate("firebase-key.json")
        except FileNotFoundError:
            cred = None
    
    if cred and not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    return cred is not None

def get_gemini_api_key():
    """
    Gemini API anahtarını güvenli bir şekilde alır.
    """
    try:
        return st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return st.sidebar.text_input("Gemini API Anahtarınızı Girin", type="password", help="Bu sadece yerel testler için gereklidir.")


# --- TÜM ANALİZ FONKSİYONLARI ---

def calistir_analiz(veri_df):
    """
    Tüm finansal metrikleri hesaplar.
    """
    if veri_df.empty: return {"hata": "Veri bulunamadı."}
    try:
        toplam_gelir = veri_df['Gelir'].sum()
        toplam_gider = veri_df['Gider'].sum()
        net_kar = toplam_gelir - toplam_gider
        gider_kategorileri = veri_df.groupby('Kategori')['Gider'].sum()
        en_yuksek_gider_kategorisi = gider_kategorileri.idxmax() if not gider_kategorileri.empty else "N/A"
        kar_marji = (net_kar / toplam_gelir * 100) if toplam_gelir > 0 else 0
        
        return {
            "toplam_gelir": toplam_gelir,
            "toplam_gider": toplam_gider,
            "net_kar": net_kar,
            "en_yuksek_gider_kategorisi": en_yuksek_gider_kategorisi,
            "kar_marji": kar_marji
        }
    except Exception as e: return {"hata": str(e)}

def prophet_tahmini_yap(aylik_veri_df):
    if len(aylik_veri_df) < 2: return None, None
    prophet_df = aylik_veri_df.reset_index().rename(columns={'Tarih': 'ds', 'Gelir': 'y'})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=3, freq='M')
    forecast = model.predict(future)
    return model, forecast

def yorum_uret(api_key, analiz_sonuclari, tahmin_trendi):
    """
    AI Danışman yorumu üretir.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Sen deneyimli, pozitif ve yol gösterici bir finansal danışmansın.
        Aşağıdaki şirket verilerini analiz ederek 2-3 paragrafta mevcut durumu özetle ve 3 maddelik somut bir eylem planı öner. Dilin profesyonel ama anlaşılır olsun.
        - Toplam Gelir: {analiz_sonuclari['toplam_gelir']:,} TL
        - Net Kar: {analiz_sonuclari['net_kar']:,} TL
        - Kar Marjı: %{analiz_sonuclari['kar_marji']:.2f}
        - En Büyük Gider Kalemi: {analiz_sonuclari['en_yuksek_gider_kategorisi']}
        - Gelecek Gelir Tahmini Trendi: {tahmin_trendi}
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        st.error("AI Yorumu üretilemedi. Lütfen API anahtarınızı kontrol edin.")
        return ""


# --- ARAYÜZ GÖSTERİM FONKSİYONLARI ---

def show_landing_page():
    # ... (Ana sayfa fonksiyonu aynı)
    st.title("Finansal Verilerinizi **Anlamlı Stratejilere** Dönüştürün")
    if st.button("🚀 Ücretsiz Denemeye Başla", type="primary"):
        st.session_state['page'] = 'login'
        st.rerun()

def show_login_page():
    # ... (Giriş/Kayıt fonksiyonu aynı)
    st.subheader("Hesabınıza Giriş Yapın veya Yeni Hesap Oluşturun")
    choice = st.radio("Seçiminiz:", ["Giriş Yap", "Kayıt Ol"], horizontal=True, label_visibility="collapsed")
    with st.form("auth_form"):
        email = st.text_input("E-posta Adresi", placeholder="ornek@mail.com")
        password = st.text_input("Şifre", type="password")
        submitted = st.form_submit_button(choice, use_container_width=True)
        if submitted:
            db = firestore.client()
            if choice == "Kayıt Ol":
                try:
                    user = auth.create_user(email=email, password=password)
                    db.collection('users').document(user.uid).set({'email': email, 'subscription_plan': 'None'})
                    st.success("Kayıt başarılı! Lütfen giriş yapın.")
                except Exception as e: st.error(f"Kayıt hatası: {e}")
            elif choice == "Giriş Yap":
                try:
                    user = auth.get_user_by_email(email)
                    st.session_state['user_info'] = {'uid': user.uid, 'email': user.email}; st.rerun()
                except Exception: st.error("E-posta adresi bulunamadı veya bir hata oluştu.")

def show_dashboard(user_info, api_key):
    """
    Kullanıcının abonelik planına göre tam özellikli analiz panelini gösterir.
    """
    db = firestore.client()
    user_doc_ref = db.collection('users').document(user_info['uid'])
    subscription_plan = user_doc_ref.get().to_dict().get('subscription_plan', 'None')

    if subscription_plan == 'None':
        st.title("Abonelik Paketleri")
        col1, col2, col3 = st.columns(3)
        if col1.button("Basic Planı Seç (₺350/ay)"): user_doc_ref.set({'subscription_plan': 'Basic'}, merge=True); st.rerun()
        if col2.button("Pro Planı Seç (₺750/ay)", type="primary"): user_doc_ref.set({'subscription_plan': 'Pro'}, merge=True); st.rerun()
        if col3.button("Enterprise Planı Seç (₺2000/ay)"): user_doc_ref.set({'subscription_plan': 'Enterprise'}, merge=True); st.rerun()
    else:
        st.title(f"🚀 Finansal Analiz Paneli")
        st.sidebar.info(f"Aktif Paketiniz: **{subscription_plan}**")
        
        uploaded_file = st.sidebar.file_uploader("CSV dosyanızı yükleyin", type="csv")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file, parse_dates=['Tarih'])
            analiz_sonuclari = calistir_analiz(df)

            if "hata" in analiz_sonuclari:
                st.error(f"Analiz hatası: {analiz_sonuclari['hata']}")
                return

            # --- TÜM METRİKLER (Tüm Paketler İçin) ---
            st.subheader("Finansal Özet")
            cols = st.columns(4)
            cols[0].metric("Toplam Gelir", f"{analiz_sonuclari.get('toplam_gelir', 0):,} TL")
            cols[1].metric("Toplam Gider", f"{analiz_sonuclari.get('toplam_gider', 0):,} TL")
            cols[2].metric("Net Kar", f"{analiz_sonuclari.get('net_kar', 0):,} TL")
            cols[3].metric("Kar Marjı", f"%{analiz_sonuclari.get('kar_marji', 0):.2f}")
            
            st.divider()

            # --- TAHMİN GRAFİĞİ VE AI YORUMU (Pro ve Enterprise Paketler İçin) ---
            if subscription_plan in ['Pro', 'Enterprise']:
                st.subheader("Gelecek Tahmini ve AI Danışman Yorumu")
                
                aylik_veri = df.set_index('Tarih')[['Gelir']].resample('M').sum()
                model, tahmin = prophet_tahmini_yap(aylik_veri)
                
                if model and tahmin is not None:
                    yorum_col, grafik_col = st.columns([1.2, 1.8])
                    
                    with grafik_col:
                        fig = plot_plotly(model, tahmin, xlabel="Tarih", ylabel="Gelir")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with yorum_col:
                        if api_key:
                            with st.spinner("AI yorum üretiyor..."):
                                trend = "Yükselişte" if tahmin['yhat'].iloc[-1] > tahmin['yhat'].iloc[-4] else "Düşüşte/Durgun"
                                yorum = yorum_uret(api_key, analiz_sonuclari, trend)
                                st.markdown(yorum)
                        else:
                            st.warning("AI yorumunu görmek için lütfen kenar çubuğundan geçerli bir API anahtarı girin.")
                else:
                    st.warning("Tahmin oluşturmak için yeterli veri yok (en az 2 ay gerekir).")
            else: # Basic Paket
                st.subheader("Aylık Gelir Trendi")
                aylik_veri = df.set_index('Tarih')[['Gelir']].resample('M').sum()
                st.line_chart(aylik_veri)
                st.info("AI Danışman Yorumu ve detaylı tahmin grafiği 'Pro' ve 'Enterprise' paketlerinde mevcuttur.")
        else:
            st.info("Lütfen analize başlamak için kenar çubuğundan bir CSV dosyası yükleyin.")

# --- ANA UYGULAMA AKIŞI ---

def main():
    firebase_ok = init_firebase()

    # Session State Yönetimi
    if 'user_info' not in st.session_state: st.session_state['user_info'] = None
    if 'page' not in st.session_state: st.session_state['page'] = 'landing'

    # Kenar Çubuğu
    with st.sidebar:
        st.header("KazKaz AI")
        if st.session_state['user_info']:
            st.write(f"Hoş Geldin, {st.session_state['user_info']['email']}")
            if st.button("Çıkış Yap"): st.session_state.clear(); st.rerun()
        else:
            st.write("Finansal geleceğinize hoş geldiniz.")
        
        api_key = get_gemini_api_key()

    # Sayfa Yönlendirme
    if not firebase_ok and 'firebase_initialized' in st.session_state:
        st.error("Uygulama başlatılamıyor. Firebase yapılandırmasını kontrol edin.")
    elif st.session_state['user_info']:
        show_dashboard(st.session_state['user_info'], api_key)
    elif st.session_state['page'] == 'login':
        show_login_page()
    else: # 'landing' veya varsayılan
        show_landing_page()

if __name__ == '__main__':
    main()
