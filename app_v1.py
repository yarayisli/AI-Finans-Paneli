import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, auth, firestore
from prophet import Prophet
from prophet.plot import plot_plotly
import requests
import google.generativeai as genai

# --- Firebase Bağlantısı (Sadece bir kere yapılır) ---
# Bu bölüm, anahtarınızı Streamlit'in gizli kasasından okur.
try:
    firebase_creds_dict = {
      "type": st.secrets["firebase"]["type"],
      "project_id": st.secrets["firebase"]["project_id"],
      "private_key_id": st.secrets["firebase"]["private_key_id"],
      "private_key": st.secrets["firebase"]["private_key"].replace('\\n', '\n'),
      "client_email": st.secrets["firebase"]["client_email"],
      "client_id": st.secrets["firebase"]["client_id"],
      "auth_uri": st.secrets["firebase"]["auth_uri"],
      "token_uri": st.secrets["firebase"]["token_uri"],
      "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
      "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"]
    }
    cred = credentials.Certificate(firebase_creds_dict)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
except (KeyError, ValueError):
    st.error("Firebase yapılandırması eksik veya hatalı. Lütfen Streamlit Cloud Secrets'ı kontrol edin.")
    st.stop()


db = firestore.client()

# --- Tüm Analiz Fonksiyonları (Değişiklik yok) ---
# calistir_analiz, doviz_kuru_getir, prophet_tahmini_yap, yorum_uret...
def calistir_analiz(veri_df):
    if veri_df.empty: return {"hata": "Filtrelenen veri bulunamadı."}
    try:
        toplam_gelir = veri_df['Gelir'].sum(); toplam_gider = veri_df['Gider'].sum(); net_kar = toplam_gelir - toplam_gider
        gider_kategorileri = veri_df.groupby('Kategori')['Gider'].sum()
        en_yuksek_gider_kategorisi = gider_kategorileri.idxmax() if not gider_kategorileri.empty else "N/A"
        en_yuksek_gider_tutari = gider_kategorileri.max() if not gider_kategorileri.empty else 0
        return {"toplam_gelir": toplam_gelir, "toplam_gider": toplam_gider, "net_kar": net_kar, "en_yuksek_gider_kategorisi": en_yuksek_gider_kategorisi, "en_yuksek_gider_tutari": en_yuksek_gider_tutari}
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
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Sen deneyimli bir finansal danışmansın. Aşağıdaki verilere dayanarak, şirketin durumu hakkında 'içten, sıcak ve insani' bir tonda, 2-3 paragraflık bir durum değerlendirmesi ve 2-3 maddelik bir eylem planı önerisi yaz.
        - Veriler: Toplam Gelir: {analiz_sonuclari['toplam_gelir']:,} TL, Net Kar: {analiz_sonuclari['net_kar']:,} TL, En Büyük Gider Kalemi: {analiz_sonuclari['en_yuksek_gider_kategorisi']}, Tahmin Trendi: {tahmin_ozeti}
        Yorumuna "Değerli Yönetici," diye başla.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"Yorum üretilirken bir hata oluştu: {e}"

# --- ANA UYGULAMA MANTIĞI ---

def main():
    st.set_page_config(page_title="AI Finans Danışmanı", layout="wide")

    # Kullanıcı giriş yapmış mı kontrol et
    if 'user_info' not in st.session_state:
        st.session_state['user_info'] = None

    if st.session_state['user_info']:
        # KULLANICI GİRİŞ YAPMIŞSA
        user_uid = st.session_state['user_info']['uid']
        user_email = st.session_state['user_info']['email']

        st.sidebar.subheader(f"Hoş Geldin, {user_email}")
        if st.sidebar.button("Çıkış Yap"):
            st.session_state['user_info'] = None
            st.rerun()

        # Abonelik durumunu veritabanından kontrol et
        user_doc_ref = db.collection('users').document(user_uid)
        user_doc = user_doc_ref.get()
        subscription_plan = user_doc.to_dict().get('subscription_plan', 'None')

        if subscription_plan == 'None':
            # FİYATLANDIRMA EKRANI
            st.title("Size Özel Abonelik Paketleri")
            st.write("Lütfen devam etmek için bir paket seçin.")
            col1, col2, col3 = st.columns(3)
            if col1.button("Basic Paket Seç (₺350/ay)"):
                user_doc_ref.set({'subscription_plan': 'Basic'}, merge=True); st.rerun()
            if col2.button("Pro Paket Seç (₺750/ay)"):
                user_doc_ref.set({'subscription_plan': 'Pro'}, merge=True); st.rerun()
            if col3.button("Enterprise Paket Seç (₺2000/ay)"):
                user_doc_ref.set({'subscription_plan': 'Enterprise'}, merge=True); st.rerun()
        else:
            # ABONELİK VARSA, ANALİZ PANELİNİ GÖSTER
            st.title(f"🚀 Finansal Analiz Paneli ({subscription_plan} Paket)")
            
            uploaded_file = st.file_uploader("Analiz için CSV dosyanızı buraya yükleyin", type="csv")
            if uploaded_file:
                ana_veri = pd.read_csv(uploaded_file, parse_dates=['Tarih'])
                
                analiz_sonuclari = calistir_analiz(ana_veri)
                if "hata" not in analiz_sonuclari:
                    st.header("Genel Finansal Durum")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Toplam Gelir", f"{analiz_sonuclari['toplam_gelir']} TL")
                    col2.metric("Toplam Gider", f"{analiz_sonuclari['toplam_gider']} TL")
                    col3.metric("Net Kar", f"{analiz_sonuclari['net_kar']} TL")
                    
                    st.divider()
                    st.header("Gelir Tahmini")
                    aylik_veri = ana_veri.set_index('Tarih')[['Gelir']].resample('ME').sum()
                    model, tahmin = prophet_tahmini_yap(aylik_veri)

                    # ABONELİK PAKETİNE GÖRE ÖZELLİK KONTROLÜ
                    if subscription_plan in ['Pro', 'Enterprise']:
                        if model and tahmin is not None:
                            fig = plot_plotly(model, tahmin, xlabel="Tarih", ylabel="Gelir")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.divider()
                            st.header("🤖 AI Danışman Yorumu")
                            with st.spinner("Yapay zeka danışmanınız verileri analiz ediyor..."):
                                son_gercek_gelir = tahmin['yhat'].iloc[-4]
                                son_tahmin_gelir = tahmin['yhat'].iloc[-1]
                                tahmin_trendi = "Yükselişte" if son_tahmin_gelir > son_gercek_gelir else "Düşüşte veya Durgun"
                                yorum = yorum_uret(st.secrets["GEMINI_API_KEY"], analiz_sonuclari, tahmin_trendi)
                                st.markdown(yorum)
                        else:
                            st.warning("Tahmin ve yorum oluşturmak için yeterli veri bulunmuyor.")
                    else: # Basic paket ise
                        st.info("AI Danışman Yorumu ve detaylı tahmin grafiği 'Pro' ve 'Enterprise' paketlerinde mevcuttur.")

    else:
        # KULLANICI GİRİŞ YAPMAMIŞSA
        choice = st.selectbox("Giriş Yap / Kayıt Ol", ["Giriş Yap", "Kayıt Ol"])
        email = st.text_input("E-posta Adresi")
        password = st.text_input("Şifre", type="password")

        if choice == "Giriş Yap":
            if st.button("Giriş Yap", type="primary"):
                try:
                    user = auth.get_user_by_email(email)
                    # Gerçekte şifre doğrulaması için Firebase'in client-side SDK'ları kullanılır.
                    # Bu prototipte girişin başarılı olduğunu varsayıyoruz.
                    st.session_state['user_info'] = {'uid': user.uid, 'email': user.email}
                    st.rerun()
                except Exception as e:
                    st.error("E-posta veya şifre hatalı.")
        else: # Kayıt Ol
            if st.button("Kayıt Ol", type="primary"):
                try:
                    user = auth.create_user(email=email, password=password)
                    db.collection('users').document(user.uid).set({'email': email, 'subscription_plan': 'None'})
                    st.success("Kayıt başarılı! Lütfen giriş yapın.")
                except Exception as e:
                    st.error(f"Kayıt sırasında bir hata oluştu: {e}")

if __name__ == '__main__':
    main()
