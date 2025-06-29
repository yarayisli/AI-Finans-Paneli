import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, auth, firestore
from prophet import Prophet
from prophet.plot import plot_plotly
import requests
import google.generativeai as genai

# --- Firebase Bağlantısı (Sadece bir kere yapılır) ---
try:
    # Bu bölüm, anahtarınızı Streamlit'in gizli kasasından okur.
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
    # Yerelde çalışırken bu hatayı görmezden gel, anahtar dosyasını kullan
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate("firebase-key.json")
            firebase_admin.initialize_app(cred)
    except FileNotFoundError:
        st.error("Firebase anahtar dosyası bulunamadı ve Streamlit Secrets ayarlanmamış.")
        st.stop()


db = firestore.client()

# --- Tüm Analiz Fonksiyonları (Değişiklik yok) ---
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
    except Exception as e:
        # Gerçek hatayı loglamak daha iyi olur, ama kullanıcıya genel bir mesaj gösterelim.
        st.error(f"AI Yorumu üretilemedi. API anahtarınızın geçerli olduğundan ve limitleri aşmadığınızdan emin olun.")
        return ""

# --- ANA UYGULAMA MANTIĞI ---

def main():
    st.set_page_config(page_title="AI Finans Danışmanı", layout="wide")

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

        user_doc_ref = db.collection('users').document(user_uid)
        user_doc = user_doc_ref.get()
        subscription_plan = user_doc.to_dict().get('subscription_plan', 'None')

        if subscription_plan == 'None':
            # FİYATLANDIRMA EKRANI
            st.title("Size Özel Abonelik Paketleri")
            # ... (Fiyatlandırma kodları aynı)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Basic"); st.write("Raporlama + özet"); st.write("₺350/ay")
                if st.button("Basic Paket Seç"): user_doc_ref.set({'subscription_plan': 'Basic'}, merge=True); st.rerun()
            with col2:
                st.subheader("Pro"); st.write("AI öneri + rapor"); st.write("₺750/ay")
                if st.button("Pro Paket Seç"): user_doc_ref.set({'subscription_plan': 'Pro'}, merge=True); st.rerun()
            with col3:
                st.subheader("Enterprise"); st.write("Çoklu kullanıcı + destek"); st.write("₺2000/ay")
                if st.button("Enterprise Paket Seç"): user_doc_ref.set({'subscription_plan': 'Enterprise'}, merge=True); st.rerun()
        else:
            # ABONELİK VARSA, ANALİZ PANELİNİ GÖSTER (DOLDURULMUŞ HALİ)
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
                                son_gercek_gelir = tahmin['yhat'].iloc[-4]; son_tahmin_gelir = tahmin['yhat'].iloc[-1]
                                tahmin_trendi = "Yükselişte" if son_tahmin_gelir > son_gercek_gelir else "Düşüşte veya Durgun"
                                yorum = yorum_uret(st.secrets["GEMINI_API_KEY"], analiz_sonuclari, tahmin_trendi)
                                st.markdown(yorum)
                        else: st.warning("Tahmin ve yorum oluşturmak için yeterli veri bulunmuyor.")
                    else: # Basic paket ise
                        st.line_chart(aylik_veri) # Sadece temel grafiği göster
                        st.info("AI Danışman Yorumu ve detaylı tahmin grafiği 'Pro' ve 'Enterprise' paketlerinde mevcuttur.")
            else:
                 st.info("Lütfen bir CSV dosyası yükleyerek analize başlayın.")

    else:
        # KULLANICI GİRİŞ YAPMAMIŞSA
        # ... (Giriş/Kayıt kodları aynı)
        choice = st.selectbox("Giriş Yap / Kayıt Ol", ["Giriş Yap", "Kayıt Ol"])
        email = st.text_input("E-posta Adresi")
        password = st.text_input("Şifre", type="password")
        if choice == "Giriş Yap":
            if st.button("Giriş Yap", type="primary"):
                try:
                    # Bu prototipte, Firebase'in şifreyi doğrulamadığını unutmayın.
                    user = auth.get_user_by_email(email)
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

        user_doc_ref = db.collection('users').document(user_uid)
        user_doc = user_doc_ref.get()
        subscription_plan = user_doc.to_dict().get('subscription_plan', 'None')

        if subscription_plan == 'None':
            # FİYATLANDIRMA EKRANI
            st.title("Size Özel Abonelik Paketleri")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Basic"); st.write("Raporlama + özet"); st.write("₺350/ay")
                if st.button("Basic Paket Seç"): user_doc_ref.set({'subscription_plan': 'Basic'}, merge=True); st.rerun()
            with col2:
                st.subheader("Pro"); st.write("AI öneri + rapor"); st.write("₺750/ay")
                if st.button("Pro Paket Seç"): user_doc_ref.set({'subscription_plan': 'Pro'}, merge=True); st.rerun()
            with col3:
                st.subheader("Enterprise"); st.write("Çoklu kullanıcı + destek"); st.write("₺2000/ay")
                if st.button("Enterprise Paket Seç"): user_doc_ref.set({'subscription_plan': 'Enterprise'}, merge=True); st.rerun()
        else:
            # ABONELİK VARSA, ANALİZ PANELİNİ GÖSTER
            st.title(f"🚀 Finansal Analiz Paneli ({subscription_plan} Paket)")
            
            # DEĞİŞİKLİK: Filtreleme başlığını dosya yükleme mantığının dışına taşıdık
            st.sidebar.divider()
            st.sidebar.header("Filtreleme Seçenekleri")
            
            uploaded_file = st.file_uploader("Analiz için CSV dosyanızı buraya yükleyin", type="csv")
            
            if uploaded_file:
                ana_veri = pd.read_csv(uploaded_file, parse_dates=['Tarih'])
                
                # DEĞİŞİKLİK: Filtreleme seçenekleri artık dosya yüklendikten sonra başlığın altında beliriyor
                urun_listesi = ["Tümü"] + sorted(ana_veri['Satilan_Urun_Adi'].unique().tolist())
                secilen_urun = st.selectbox("Ürüne Göre Filtrele:", urun_listesi)

                if secilen_urun == "Tümü":
                    filtrelenmis_veri = ana_veri
                else:
                    filtrelenmis_veri = ana_veri[ana_veri['Satilan_Urun_Adi'] == secilen_urun]
                
                # --- Analiz ve Gösterge Paneli ---
                # ... (Geri kalan tüm analiz, metrik ve grafik kodları aynı kalacak)
                analiz_sonuclari = calistir_analiz(filtrelenmis_veri)
                if "hata" not in analiz_sonuclari:
                    st.header(f"'{secilen_urun}' için Finansal Durum")
                    # ... metrikler ...
                    st.divider()
                    st.header(f"'{secilen_urun}' için Profesyonel Gelir Tahmini")
                    aylik_veri = filtrelenmis_veri.set_index('Tarih')[['Gelir']].resample('ME').sum()
                    model, tahmin = prophet_tahmini_yap(aylik_veri)
                    if subscription_plan in ['Pro', 'Enterprise']:
                        if model and tahmin is not None:
                            # ... AI yorumu ve grafiği ...
                            fig = plot_plotly(model, tahmin, xlabel="Tarih", ylabel="Gelir")
                            st.plotly_chart(fig, use_container_width=True)
                            st.divider()
                            st.header("🤖 AI Danışman Yorumu")
                            # ... yorum üretme kodu ...
                        else: st.warning("Tahmin için yeterli veri yok.")
                    else:
                        st.line_chart(aylik_veri)
                        st.info("AI Danışman Yorumu 'Pro' paketinde mevcuttur.")
            else:
                 st.info("Lütfen bir CSV dosyası yükleyerek analize başlayın.")
    else:
        # KULLANICI GİRİŞ YAPMAMIŞSA
        # ... (Giriş/Kayıt kodları aynı)
        choice = st.selectbox("Giriş Yap / Kayıt Ol", ["Giriş Yap", "Kayıt Ol"])
        # ...
        
# Bu satır en altta kalacak
if __name__ == '__main__':
    main()
