# KazKaz Finansal Danışman - Nihai ve Hata Düzeltilmiş Sürüm
import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, auth, firestore
from prophet import Prophet
from prophet.plot import plot_plotly
import google.generativeai as genai

# --- Sayfa Yapılandırması ve Stil (En başta bir kere yapılır) ---
st.set_page_config(page_title="KazKaz Finansal Danışman", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
    /* Ana Gövde ve Fontlar */
    body { font-family: 'Segoe UI', sans-serif; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; padding-left: 2rem; padding-right: 2rem; }
    /* Kenar çubuğu (sidebar) stilini özelleştirme */
    .st-emotion-cache-16txtl3 { background-color: #0f172a; }
    /* Butonlar */
    .stButton > button {
        border-radius: 8px; border: 2px solid #10b981; color: white; background-color: #10b981;
        transition: all 0.3s; font-weight: bold; padding: 10px 24px; width: 100%;
    }
    .stButton > button:hover {
        border-color: #34d399; color: white; background-color: #34d399;
    }
    /* Metrik Kutucukları */
    .st-emotion-cache-1gulkj5 { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; }
    /* Başlıklar */
    h1 { font-size: 3rem; font-weight: 900; color: #ffffff; }
    h2 { font-size: 2.25rem; font-weight: 700; color: #ffffff; }
    h3 { font-size: 1.5rem; font-weight: 600; color: #ffffff; }
    /* Giriş/Kayıt Alanları */
    .stTextInput > div > div > input { background-color: #1e293b; color: white; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# --- GÜVENLİ BAĞLANTI VE ANAHTAR YÖNETİMİ ---

@st.cache_resource
def init_firebase():
    """
    Firebase bağlantısını güvenli bir şekilde başlatır.
    Bu fonksiyonun @st.cache_resource ile işaretlenmesi, bağlantının sadece bir kere kurulmasını sağlar.
    """
    try:
        # Önce Streamlit Cloud'daki gizli kasayı dener
        firebase_creds_dict = st.secrets["firebase"]
        cred = credentials.Certificate(firebase_creds_dict)
    except (KeyError, FileNotFoundError):
        # Eğer bulamazsa, yereldeki anahtar dosyasını dener
        try:
            cred = credentials.Certificate("firebase-key.json")
        except FileNotFoundError:
            return None # İki yöntem de başarısız olursa, bağlantı kurma
    
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    return True

def get_gemini_api_key():
    """
    Gemini API anahtarını güvenli bir şekilde alır.
    Önce buluttaki kasayı, sonra kenar çubuğunu dener.
    """
    try:
        # Önce Streamlit Cloud'daki gizli kasayı dener
        return st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        # Eğer bulamazsa, yerelde kullanıcıdan ister
        return st.sidebar.text_input("Gemini API Anahtarınızı Girin", type="password", help="Bu anahtar sadece yerel testler için gereklidir.")


# --- TÜM ANALİZ FONKSİYONLARI ---

def calistir_analiz(veri_df):
    if veri_df.empty: return {"hata": "Filtrelenen veri bulunamadı."}
    try:
        toplam_gelir = veri_df['Gelir'].sum()
        toplam_gider = veri_df['Gider'].sum()
        net_kar = toplam_gelir - toplam_gider
        gider_kategorileri = veri_df.groupby('Kategori')['Gider'].sum()
        en_yuksek_gider_kategorisi = gider_kategorileri.idxmax() if not gider_kategorileri.empty else "N/A"
        kar_marji = (net_kar / toplam_gelir * 100) if toplam_gelir > 0 else 0
        return {"toplam_gelir": toplam_gelir, "toplam_gider": toplam_gider, "net_kar": net_kar, "en_yuksek_gider_kategorisi": en_yuksek_gider_kategorisi, "kar_marji": kar_marji}
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
        Veriler: Toplam Gelir: {analiz_sonuclari['toplam_gelir']:,} TL, Net Kar: {analiz_sonuclari['net_kar']:,} TL, Kar Marjı: %{analiz_sonuclari['kar_marji']:.2f}, En Büyük Gider Kalemi: {analiz_sonuclari['en_yuksek_gider_kategorisi']}, Tahmin Trendi: {tahmin_trendi}
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"AI Yorumu üretilemedi: {e}")
        return ""


# --- ARAYÜZ GÖSTERİM FONKSİYONLARI ---

def show_landing_page():
    st.title("Finansal Verilerinizi **Anlamlı Stratejilere** Dönüştürün")
    st.subheader("KazKaz AI, işletmenizin finansal sağlığını analiz eder, geleceği tahminler ve size özel eylem planları sunar.")
    if st.button("🚀 Ücretsiz Denemeye Başla", type="primary"):
        st.session_state['page'] = 'login'
        st.rerun()

def show_login_page(db):
    st.subheader("Hesabınıza Giriş Yapın veya Yeni Hesap Oluşturun")
    choice = st.radio("Seçiminiz:", ["Giriş Yap", "Kayıt Ol"], horizontal=True, label_visibility="collapsed")
    with st.form("auth_form"):
        email = st.text_input("E-posta Adresi", placeholder="ornek@mail.com")
        password = st.text_input("Şifre", type="password")
        submitted = st.form_submit_button(choice, use_container_width=True)
        if submitted:
            if choice == "Kayıt Ol":
                try:
                    user = auth.create_user(email=email, password=password)
                    db.collection('users').document(user.uid).set({'email': email, 'subscription_plan': 'None'})
                    st.success("Kayıt başarılı! Lütfen giriş yapın.")
                except Exception as e: st.error(f"Kayıt hatası: {e}")
            elif choice == "Giriş Yap":
                try:
                    user = auth.get_user_by_email(email)
                    st.session_state['user_info'] = {'uid': user.uid, 'email': user.email}
                    st.rerun()
                except Exception: st.error("E-posta adresi bulunamadı veya bir hata oluştu.")

def show_dashboard(user_info, api_key):
    db = firestore.client()
    user_doc_ref = db.collection('users').document(user_info['uid'])
    subscription_plan = user_doc_ref.get().to_dict().get('subscription_plan', 'None')

    if subscription_plan == 'None':
        st.title("Abonelik Paketleri")
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            st.subheader("Basic")
            st.metric(label="Raporlama + Özet", value="₺350", delta="/aylık")
            if st.button("Basic Planı Seç"): user_doc_ref.set({'subscription_plan': 'Basic'}, merge=True); st.rerun()
        with p_col2:
            st.subheader("Pro")
            st.metric(label="AI Öneri + Rapor", value="₺750", delta="/aylık")
            if st.button("Pro Planı Seç", type="primary"): user_doc_ref.set({'subscription_plan': 'Pro'}, merge=True); st.rerun()
        with p_col3:
            st.subheader("Enterprise")
            st.metric(label="Çoklu Kullanıcı + Destek", value="₺2000", delta="/aylık")
            if st.button("Enterprise Planı Seç"): user_doc_ref.set({'subscription_plan': 'Enterprise'}, merge=True); st.rerun()
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
    if 'user_info' not in st.session_state: st.session_state['user_info'] = None
    if 'page' not in st.session_state: st.session_state['page'] = 'landing'
    
    firebase_ok = init_firebase()

    # Kenar Çubuğu
    with st.sidebar:
        st.header("KazKaz AI")
        if st.session_state['user_info']:
            st.write(f"Hoş Geldin, {st.session_state['user_info']['email']}")
            if st.button("Çıkış Yap"):
                st.session_state.clear()
                st.rerun()
        else:
            st.write("Finansal geleceğinize hoş geldiniz.")
    
    # Sayfa Yönlendirme
    if not firebase_ok:
        st.warning("Firebase bağlantısı kurulamadı. Lütfen `firebase-key.json` dosyanızın veya Streamlit Secrets ayarlarınızın doğru olduğundan emin olun.")
    elif st.session_state['user_info']:
        api_key = get_gemini_api_key()
        show_dashboard(st.session_state['user_info'], api_key)
    elif st.session_state['page'] == 'login':
        # Giriş/Kayıt sayfasında db'ye ihtiyacımız var
        if firebase_ok:
            db = firestore.client()
            show_login_page(db)
    else:
        show_landing_page()

if __name__ == '__main__':
    main()
