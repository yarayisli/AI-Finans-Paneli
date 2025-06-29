import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, auth, firestore
from prophet import Prophet
from prophet.plot import plot_plotly
import requests
import google.generativeai as genai

# --- Sayfa Yapılandırması ve Stil Enjeksiyonu ---
st.set_page_config(page_title="KazKaz Finansal Danışman", layout="wide", initial_sidebar_state="expanded")

# --- HEDEF SİTEYE BENZEMESİ İÇİN CSS STİLLERİ ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Bu CSS'i bir dosyaya koymak yerine doğrudan enjekte ediyoruz
st.markdown("""
<style>
    /* Buton Stilleri */
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
    /* Metrik Kutucukları */
    .st-emotion-cache-1gulkj5 { /* Streamlit'in metrik konteyner class'ı */
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
    }
    /* Başlıklar */
    h1, h2, h3 {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# --- Firebase Bağlantısı (Sadece bir kere yapılır) ---
try:
    firebase_creds_dict = st.secrets["firebase"]
    cred = credentials.Certificate(firebase_creds_dict)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
except Exception:
    if not firebase_admin._apps:
        try:
            cred = credentials.Certificate("firebase-key.json")
            firebase_admin.initialize_app(cred)
        except FileNotFoundError:
            pass # Sadece ilk çalıştırmada hata vermemesi için

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
    st.title("Finansal Özgürlüğe Ulaşmanın En Kolay Yolu")
    st.subheader("KazKaz Finansal Danışman ile verilerinizi anlamlı bilgilere ve stratejilere dönüştürün.")
    
    st.write(" ") # Boşluk için
    if st.button("Hemen Başla", type="primary"):
        # Kullanıcıyı giriş/kayıt bölümüne yönlendirmek için bir state değişikliği
        st.session_state['page'] = 'login'
        st.rerun()
    
    st.divider()
    
    st.header("Özelliklerimiz")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📊 Kapsamlı Raporlama")
        st.write("Gelir, gider ve kar metriklerinizi anlık olarak takip edin.")
    with col2:
        st.subheader("🤖 AI Destekli Öngörü")
        st.write("Gelecekteki gelir trendlerinizi profesyonel tahmin modelleri ile görün.")
    with col3:
        st.subheader("💡 Kişiye Özel Tavsiye")
        st.write("Yapay zeka danışmanınızdan size özel, anlaşılır eylem planları alın.")

    # Fiyatlandırmayı ana sayfada da gösterelim
    st.divider()
    st.header("Paketlerimiz")
    p_col1, p_col2, p_col3 = st.columns(3)
    p_col1.metric(label="Basic", value="₺350/ay", delta="Raporlama + Özet")
    p_col2.metric(label="Pro", value="₺750/ay", delta="AI Öneri + Rapor")
    p_col3.metric(label="Enterprise", value="₺2000/ay", delta="Çoklu Kullanıcı + Destek")

def show_dashboard(subscription_plan):
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
            st.header("Gelir Tahmini")
            aylik_veri = ana_veri.set_index('Tarih')[['Gelir']].resample('ME').sum()
            model, tahmin = prophet_tahmini_yap(aylik_veri)
            if subscription_plan in ['Pro', 'Enterprise']:
                if model and tahmin is not None:
                    fig = plot_plotly(model, tahmin, xlabel="Tarih", ylabel="Gelir")
                    st.plotly_chart(fig, use_container_width=True)
                    st.divider()
                    st.header("🤖 AI Danışman Yorumu")
                    with st.spinner("Yapay zeka danışmanınız verileri analiz ediyor..."):
                        tahmin_trendi = "Yükselişte" if tahmin['yhat'].iloc[-1] > tahmin['yhat'].iloc[-4] else "Düşüşte veya Durgun"
                        yorum = yorum_uret(st.secrets["GEMINI_API_KEY"], analiz_sonuclari, tahmin_trendi)
                        st.markdown(yorum)
                else: st.warning("Tahmin için yeterli veri yok.")
            else:
                st.line_chart(aylik_veri)
                st.info("AI Danışman Yorumu 'Pro' paketinde mevcuttur.")
    else: st.info("Lütfen bir CSV dosyası yükleyerek analize başlayın.")

def main():
    if 'user_info' not in st.session_state: st.session_state['user_info'] = None
    if 'page' not in st.session_state: st.session_state['page'] = 'landing'

    # Giriş yapmış kullanıcı kontrolü
    if st.session_state['user_info']:
        user_uid = st.session_state['user_info']['uid']
        user_email = st.session_state['user_info']['email']
        st.sidebar.subheader(f"Hoş Geldin, {user_email}")
        if st.sidebar.button("Çıkış Yap"):
            st.session_state['user_info'] = None
            st.session_state['page'] = 'landing' # Çıkış yapınca ana sayfaya dön
            st.rerun()
        
        db = firestore.client()
        user_doc_ref = db.collection('users').document(user_uid)
        user_doc = user_doc_ref.get()
        subscription_plan = user_doc.to_dict().get('subscription_plan', 'None')
        
        if subscription_plan == 'None':
            st.title("Abonelik Paketleri")
            col1, col2, col3 = st.columns(3)
            if col1.button("Basic Paket Seç (₺350/ay)"): user_doc_ref.set({'subscription_plan': 'Basic'}, merge=True); st.rerun()
            if col2.button("Pro Paket Seç (₺750/ay)"): user_doc_ref.set({'subscription_plan': 'Pro'}, merge=True); st.rerun()
            if col3.button("Enterprise Paket Seç (₺2000/ay)"): user_doc_ref.set({'subscription_plan': 'Enterprise'}, merge=True); st.rerun()
        else:
            show_dashboard(subscription_plan)
    else:
        # GİRİŞ YAPMAMIŞ KULLANICI
        if st.session_state['page'] == 'landing':
            show_landing_page()
        else: # Giriş/Kayıt ekranı
            choice = st.selectbox("Giriş Yap / Kayıt Ol", ["Giriş Yap", "Kayıt Ol"])
            email = st.text_input("E-posta Adresi")
            password = st.text_input("Şifre", type="password")
            db = firestore.client()
            if choice == "Giriş Yap":
                if st.button("Giriş Yap", type="primary"):
                    try:
                        user = auth.get_user_by_email(email)
                        st.session_state['user_info'] = {'uid': user.uid, 'email': user.email}
                        st.rerun()
                    except: st.error("E-posta veya şifre hatalı.")
            else:
                if st.button("Kayıt Ol", type="primary"):
                    try:
                        user = auth.create_user(email=email, password=password)
                        db.collection('users').document(user.uid).set({'email': email, 'subscription_plan': 'None'})
                        st.success("Kayıt başarılı! Lütfen giriş yapın.")
                    except Exception as e: st.error(f"Kayıt hatası: {e}")

if __name__ == '__main__':
    main()
