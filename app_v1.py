# KazKaz Finansal Danışman - Profesyonel Site Versiyonu
import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, auth, firestore
from prophet import Prophet
from prophet.plot import plot_plotly
import google.generativeai as genai

# --- Sayfa Yapılandırması (En başta bir kere yapılır) ---
st.set_page_config(page_title="KazKaz Finansal Danışman", layout="wide", initial_sidebar_state="collapsed")

# --- CSS Stilleri (Sitenin görünümünü tamamen değiştirir) ---
st.markdown("""
<style>
    /* Ana Gövde ve Fontlar */
    body { font-family: 'Segoe UI', sans-serif; }
    /* Streamlit'in ana bloğunu tam genişlik yapmak */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    /* Kenar çubuğu (sidebar) stilini özelleştirme */
    .st-emotion-cache-16txtl3 { background-color: #0f172a; }
    /* Butonlar */
    .stButton > button {
        border-radius: 8px; border: 2px solid #10b981; color: white; background-color: #10b981;
        transition: all 0.3s; font-weight: bold; padding: 10px 24px;
    }
    .stButton > button:hover {
        border-color: #34d399; color: white; background-color: #34d399;
    }
    /* Metrik Kutucukları */
    .st-emotion-cache-1gulkj5 { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; }
    /* Başlıklar */
    h1 { font-size: 3rem; font-weight: 900; }
    h2 { font-size: 2.25rem; font-weight: 700; }
    h3 { font-size: 1.5rem; font-weight: 600; }
    /* Giriş/Kayıt Alanları */
    .stTextInput > div > div > input { background-color: #1e293b; color: white; border-radius: 8px; }
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
        cred = None # Anahtar bulunamazsa devam et, hata verme

if cred and not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# --- Tüm Analiz Fonksiyonları (Değişiklik yok) ---
# calistir_analiz, prophet_tahmini_yap, yorum_uret fonksiyonları burada...
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


# --- ARAYÜZ GÖSTERİM FONKSİYONLARI ---

def show_landing_page():
    # --- Hero Section ---
    st.title("Finansal Verilerinizi **Anlamlı Stratejilere** Dönüştürün")
    st.subheader("KazKaz AI, işletmenizin finansal sağlığını analiz eder, geleceği tahminler ve size özel eylem planları sunar.")
    st.write(" ")
    if st.button("🚀 Ücretsiz Denemeye Başla"):
        st.session_state['page'] = 'login'
        st.rerun()

    st.divider()

    # --- Özellikler Section ---
    st.header("Neden KazKaz AI?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📊 Otomatik Raporlama")
        st.write("Gelir, gider ve kar metriklerinizi anlık olarak takip edin. Karmaşık tablolara veda edin.")
    with col2:
        st.subheader("🤖 Akıllı Öngörüler")
        st.write("Gelecekteki gelir trendlerinizi endüstri standardı tahmin modelleri ile görerek bir adım önde olun.")
    with col3:
        st.subheader("💡 Eyleme Geçirilebilir Tavsiyeler")
        st.write("Yapay zeka danışmanınızdan size özel, net ve anlaşılır eylem planları alın.")

    st.divider()

    # --- Fiyatlandırma Section ---
    st.header("Size Uygun Bir Plan Mutlaka Var")
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        st.subheader("Basic")
        st.metric(label="Raporlama + Özet", value="₺350", delta="/aylık")
        st.button("Basic Planı Seç", key="landing_basic", on_click=lambda: st.session_state.update({'page': 'login'}))
    with p_col2:
        st.subheader("Pro")
        st.metric(label="AI Öneri + Rapor", value="₺750", delta="/aylık")
        st.button("Pro Planı Seç", key="landing_pro", type="primary", on_click=lambda: st.session_state.update({'page': 'login'}))
    with p_col3:
        st.subheader("Enterprise")
        st.metric(label="Çoklu Kullanıcı + Destek", value="₺2000", delta="/aylık")
        st.button("Enterprise Planı Seç", key="landing_enterprise", on_click=lambda: st.session_state.update({'page': 'login'}))


def show_dashboard(subscription_plan, api_key):
    st.sidebar.title("Kontrol Paneli")
    st.title(f"🚀 Finansal Analiz Paneli")
    st.sidebar.info(f"Aktif Paketiniz: **{subscription_plan}**")
    
    uploaded_file = st.sidebar.file_uploader("CSV dosyanızı yükleyin", type="csv")
    if uploaded_file:
        ana_veri = pd.read_csv(uploaded_file, parse_dates=['Tarih'])
        analiz_sonuclari = calistir_analiz(ana_veri)
        if "hata" not in analiz_sonuclari:
            st.subheader("Genel Finansal Durum")
            col1, col2, col3 = st.columns(3)
            col1.metric("Toplam Gelir", f"{analiz_sonuclari['toplam_gelir']:,} TL")
            col2.metric("Toplam Gider", f"{analiz_sonuclari['toplam_gider']:,} TL")
            col3.metric("Net Kar", f"{analiz_sonuclari['net_kar']:,} TL")
            
            st.divider()
            if subscription_plan in ['Pro', 'Enterprise']:
                st.subheader("🤖 AI Danışman Yorumu ve Tahmin")
                aylik_veri = ana_veri.set_index('Tarih')[['Gelir']].resample('ME').sum()
                model, tahmin = prophet_tahmini_yap(aylik_veri)
                if model and tahmin is not None:
                    yorum_col, grafik_col = st.columns([1, 1.5]) # Yorum solda, grafik sağda
                    with yorum_col:
                         with st.spinner("AI yorum üretiyor..."):
                            tahmin_trendi = "Yükselişte" if tahmin['yhat'].iloc[-1] > tahmin['yhat'].iloc[-4] else "Düşüşte/Durgun"
                            yorum = yorum_uret(api_key, analiz_sonuclari, tahmin_trendi)
                            st.markdown(yorum)
                    with grafik_col:
                        fig = plot_plotly(model, tahmin, xlabel="Tarih", ylabel="Gelir")
                        st.plotly_chart(fig, use_container_width=True)
                else: st.warning("Tahmin için yeterli veri yok.")
            else: # Basic
                 st.subheader("Aylık Gelir Trendi")
                 aylik_veri = ana_veri.set_index('Tarih')[['Gelir']].resample('ME').sum()
                 st.line_chart(aylik_veri)
                 st.info("AI Danışman Yorumu ve detaylı tahmin 'Pro' paketinde mevcuttur.")
    else:
        st.info("Lütfen analize başlamak için kenar çubuğundan bir CSV dosyası yükleyin.")


def main():
    # Session State Yönetimi
    if 'user_info' not in st.session_state: st.session_state['user_info'] = None
    if 'page' not in st.session_state: st.session_state['page'] = 'landing'

    firebase_ok = init_firebase()

    # Giriş yapmış kullanıcı
    if st.session_state['user_info']:
        user_uid = st.session_state['user_info']['uid']
        db = firestore.client()
        user_doc_ref = db.collection('users').document(user_uid)
        user_doc = user_doc_ref.get()
        subscription_plan = user_doc.to_dict().get('subscription_plan', 'None')
        
        # Kenar çubuğu
        st.sidebar.subheader(f"Hoş Geldin, {st.session_state['user_info']['email']}")
        if st.sidebar.button("Çıkış Yap"):
            st.session_state.clear(); st.rerun()

        if subscription_plan == 'None':
            st.title("Abonelik Paketleri"); col1, col2, col3 = st.columns(3)
            if col1.button("Basic Paket Seç"): user_doc_ref.set({'subscription_plan': 'Basic'}, merge=True); st.rerun()
            if col2.button("Pro Paket Seç", type="primary"): user_doc_ref.set({'subscription_plan': 'Pro'}, merge=True); st.rerun()
            if col3.button("Enterprise Paket Seç"): user_doc_ref.set({'subscription_plan': 'Enterprise'}, merge=True); st.rerun()
        else:
            api_key = st.secrets.get("GEMINI_API_KEY") if "GEMINI_API_KEY" in st.secrets else None
            show_dashboard(subscription_plan, api_key)
    
    # Giriş yapmamış kullanıcı
    else:
        if st.session_state['page'] == 'landing':
            show_landing_page()
        else: # Giriş/Kayıt ekranı
            st.subheader("Hesabınıza Giriş Yapın")
            db = firestore.client()
            email = st.text_input("E-posta")
            password = st.text_input("Şifre", type="password")
            if st.button("Giriş Yap", type="primary"):
                try:
                    user = auth.get_user_by_email(email)
                    st.session_state['user_info'] = {'uid': user.uid, 'email': user.email}; st.rerun()
                except: st.error("E-posta veya şifre hatalı.")
            
            if st.button("Yeni Hesap Oluştur"):
                try:
                    user = auth.create_user(email=email, password=password)
                    db.collection('users').document(user.uid).set({'email': email, 'subscription_plan': 'None'})
                    st.success("Kayıt başarılı! Lütfen giriş yapın.")
                except Exception as e: st.error(f"Kayıt hatası: {e}")

if __name__ == '__main__':
    main()
