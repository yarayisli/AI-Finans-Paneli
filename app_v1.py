# KazKaz AI Finansal Danışman - Geliştirilmiş Kimlik Doğrulama
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
    /* ... (Daha önceki CSS stillerimiz burada, değişiklik yok) ... */
    .stButton > button { border-radius: 8px; border: 2px solid #10b981; color: white; background-color: #10b981; transition: all 0.3s; font-weight: bold; padding: 10px 24px; }
    .stButton > button:hover { border-color: #34d399; color: white; background-color: #34d399; }
    h1 { font-size: 3rem; font-weight: 900; }
</style>
""", unsafe_allow_html=True)


# --- Firebase Bağlantısı (Hibrit: Yerel + Bulut) ---
# Bu fonksiyon, uygulamanın çökmemesi için en başta bir kere çalışır.
def init_firebase():
    try:
        if not firebase_admin._apps:
            try:
                # Önce Streamlit Cloud Secrets'ı dene
                firebase_creds_dict = st.secrets["firebase"]
                cred = credentials.Certificate(firebase_creds_dict)
            except (KeyError, FileNotFoundError):
                # Eğer bulamazsa, yereldeki anahtar dosyasını dene
                cred = credentials.Certificate("firebase-key.json")
            
            firebase_admin.initialize_app(cred)
        return True
    except FileNotFoundError:
        st.error("Firebase anahtar dosyası (`firebase-key.json`) bulunamadı ve Streamlit Secrets ayarlanmamış.")
        return False
    except Exception as e:
        st.error(f"Firebase başlatılırken bir hata oluştu: {e}")
        return False

# --- Tüm Analiz Fonksiyonları (Değişiklik yok) ---
# calistir_analiz, prophet_tahmini_yap, yorum_uret fonksiyonları burada...
def calistir_analiz(veri_df):
    if veri_df.empty: return {"hata": "Veri bulunamadı."}
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
        prompt = f"Sen deneyimli bir finansal danışmansın. Şu verilere dayanarak içten bir durum değerlendirmesi ve 3 maddelik eylem planı yaz: {analiz_sonuclari}, Tahmin Trendi: {tahmin_trendi}"
        response = model.generate_content(prompt); return response.text
    except Exception as e: st.error(f"AI Yorumu üretilemedi: {e}"); return ""


# --- ARAYÜZ GÖSTERİM FONKSİYONLARI ---

def show_landing_page():
    st.title("Finansal Verilerinizi **Anlamlı Stratejilere** Dönüştürün")
    st.subheader("KazKaz AI, işletmenizin finansal sağlığını analiz eder, geleceği tahminler ve size özel eylem planları sunar.")
    st.write(" ")
    if st.button("🚀 Ücretsiz Denemeye Başla", type="primary"):
        st.session_state['page'] = 'login' # Kullanıcıyı giriş sayfasına yönlendir
        st.rerun()

def show_login_page():
    st.subheader("Hesabınıza Giriş Yapın veya Yeni Hesap Oluşturun")
    choice = st.radio("Seçiminiz:", ["Giriş Yap", "Kayıt Ol"], horizontal=True)
    
    with st.form("auth_form"):
        email = st.text_input("E-posta Adresi")
        password = st.text_input("Şifre", type="password")
        submitted = st.form_submit_button(choice)

        if submitted:
            db = firestore.client()
            if choice == "Kayıt Ol":
                try:
                    user = auth.create_user(email=email, password=password)
                    db.collection('users').document(user.uid).set({'email': email, 'subscription_plan': 'None'})
                    st.success("Kayıt başarılı! Şimdi giriş yapabilirsiniz.")
                except Exception as e:
                    st.error(f"Kayıt sırasında bir hata oluştu: {e}")
            
            elif choice == "Giriş Yap":
                try:
                    # DİKKAT: Bu prototipte şifre doğrulaması yapılmamaktadır.
                    # Gerçek bir üründe bu işlem için client-side SDK'lar kullanılmalıdır.
                    user = auth.get_user_by_email(email)
                    st.session_state['user_info'] = {'uid': user.uid, 'email': user.email}
                    st.rerun() # Başarılı girişte sayfayı yenile
                except Exception as e:
                    st.error("E-posta adresi bulunamadı veya bir hata oluştu.")

def show_dashboard(user_info, api_key):
    db = firestore.client()
    user_doc_ref = db.collection('users').document(user_info['uid'])
    user_doc = user_doc_ref.get()
    subscription_plan = user_doc.to_dict().get('subscription_plan', 'None')

    if subscription_plan == 'None':
        st.title("Abonelik Paketleri")
        if st.button("Pro Paket Seç (₺750/ay)", type="primary"): 
            user_doc_ref.set({'subscription_plan': 'Pro'}, merge=True)
            st.rerun()
    else:
        st.title(f"🚀 Finansal Analiz Paneli ({subscription_plan} Paket)")
        uploaded_file = st.sidebar.file_uploader("CSV dosyanızı yükleyin", type="csv")
        if uploaded_file:
            # ... (Burada tüm analiz ve gösterim kodlarımız yer alacak)
            df = pd.read_csv(uploaded_file, parse_dates=['Tarih'])
            analiz_sonuclari = calistir_analiz(df)
            st.metric("Net Kar", f"{analiz_sonuclari.get('net_kar', 0):,} TL")
            # ... Diğer metrikler ve grafikler...
            if subscription_plan == 'Pro':
                 st.header("🤖 AI Danışman Yorumu")
                 # ... Yorum kodu...

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
            if st.button("Çıkış Yap"):
                st.session_state.clear(); st.rerun()
        else:
            st.write("Finansal geleceğinize hoş geldiniz.")

    # Sayfa Yönlendirme
    if not firebase_ok:
        st.error("Uygulama başlatılamıyor. Firebase yapılandırmasını kontrol edin.")
    elif st.session_state['user_info']:
        # KULLANICI GİRİŞ YAPMIŞSA
        api_key = st.secrets.get("GEMINI_API_KEY")
        show_dashboard(st.session_state['user_info'], api_key)
    elif st.session_state['page'] == 'login':
        show_login_page()
    else: # 'landing' veya varsayılan
        show_landing_page()

if __name__ == '__main__':
    main()
