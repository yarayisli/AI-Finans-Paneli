import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, auth, firestore

# --- Firebase Bağlantısı (Sadece bir kere yapılır) ---
try:
    # Bu, Streamlit Cloud'un gizli kasasından anahtarı okur.
    # Yerel'de çalışırken, klasörde `firebase-key.json` olmalı.
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
    firebase_admin.initialize_app(cred)
except ValueError:
    # Eğer zaten başlatıldıysa hata vermemesi için
    pass
except KeyError:
     st.error("Firebase yapılandırması eksik. Lütfen Streamlit Cloud Secrets'ı kontrol edin.")


db = firestore.client()

# --- ANA UYGULAMA ---

st.title("💡 AI Destekli Finansal Danışman")

# Session state'i kullanarak kullanıcının durumunu takip et
if 'user_info' not in st.session_state:
    st.session_state['user_info'] = None

# Kullanıcı giriş yaptıysa ana paneli göster
if st.session_state['user_info']:
    st.sidebar.subheader(f"Hoş Geldin, {st.session_state['user_info']['email']}")
    if st.sidebar.button("Çıkış Yap"):
        st.session_state['user_info'] = None
        st.rerun()

    # Abonelik durumunu kontrol et
    user_doc = db.collection('users').document(st.session_state['user_info']['uid']).get()
    subscription_plan = user_doc.to_dict().get('subscription_plan', 'None')

    if subscription_plan == 'None':
        # --- FİYATLANDIRMA EKRANI ---
        st.header("Size Özel Abonelik Paketleri")
        st.write("Lütfen devam etmek için bir paket seçin.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Basic"); st.write("Raporlama + özet"); st.write("₺350/ay")
            if st.button("Basic Paket Seç"):
                db.collection('users').document(st.session_state['user_info']['uid']).set({'subscription_plan': 'Basic'}, merge=True)
                st.success("Basic pakete abone oldunuz!")
                st.rerun()
        with col2:
            st.subheader("Pro"); st.write("AI öneri + rapor"); st.write("₺750/ay")
            if st.button("Pro Paket Seç"):
                db.collection('users').document(st.session_state['user_info']['uid']).set({'subscription_plan': 'Pro'}, merge=True)
                st.success("Pro pakete abone oldunuz!")
                st.rerun()
        with col3:
            st.subheader("Enterprise"); st.write("Çoklu kullanıcı + destek"); st.write("₺2000/ay")
            if st.button("Enterprise Paket Seç"):
                db.collection('users').document(st.session_state['user_info']['uid']).set({'subscription_plan': 'Enterprise'}, merge=True)
                st.success("Enterprise pakete abone oldunuz!")
                st.rerun()
    else:
        # --- KULLANICININ ABONELİĞİ VARSA ANALİZ PANELİNİ GÖSTER ---
        st.header(f"Aktif Paketiniz: {subscription_plan}")
        st.write("Analiz paneline hoş geldiniz!")
        # Buraya daha önce yazdığımız tüm analiz, dosya yükleme ve grafik kodları gelecek.
        # Örneğin:
        # if subscription_plan == 'Basic':
        #    st.write("Sadece Raporlama ve Özet gösterilir.")
        # elif subscription_plan == 'Pro':
        #    st.write("AI Öneri + Rapor gösterilir.")
        # elif subscription_plan == 'Enterprise':
        #    st.write("Çoklu kullanıcı ve Destek özellikleri burada yer alır.")

else:
    # --- GİRİŞ / KAYIT EKRANI ---
    choice = st.selectbox("Giriş Yap / Kayıt Ol", ["Giriş Yap", "Kayıt Ol"])
    
    email = st.text_input("E-posta Adresi")
    password = st.text_input("Şifre", type="password")

    if choice == "Giriş Yap":
        if st.button("Giriş Yap"):
            try:
                user = auth.get_user_by_email(email)
                # Not: Gerçekte şifre doğrulaması backend'de yapılır. Bu sadece bir simülasyon.
                # Gerçek bir app için Firebase'in kendi SDK'larını kullanmak gerekir.
                st.success("Giriş başarılı!")
                st.session_state['user_info'] = {'uid': user.uid, 'email': user.email}
                st.rerun()
            except Exception as e:
                st.error("E-posta veya şifre hatalı.")

    else: # Kayıt Ol
        if st.button("Kayıt Ol"):
            try:
                user = auth.create_user(email=email, password=password)
                db.collection('users').document(user.uid).set({'email': email, 'subscription_plan': 'None'})
                st.success("Kayıt başarılı! Lütfen giriş yapın.")
            except Exception as e:
                st.error(f"Kayıt sırasında bir hata oluştu: {e}")
