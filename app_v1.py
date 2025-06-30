# KazKaz AI Finansal Danışman - Gelişmiş ve Tam Donanımlı Panel
import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, auth, firestore
from prophet import Prophet
from prophet.plot import plot_plotly
import google.generativeai as genai
import plotly.graph_objects as go
import plotly.express as px

# --- Sayfa Yapılandırması ve Stil ---
st.set_page_config(page_title="KazKaz Finansal Danışman", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
    /* ... (CSS Stillerimiz aynı kalıyor ve geliştiriliyor) ... */
    .stButton > button { border-radius: 8px; border: 2px solid #10b981; color: white; background-color: #10b981; transition: all 0.3s; font-weight: bold; padding: 10px 24px; width: 100%; }
    .stButton > button:hover { border-color: #34d399; color: white; background-color: #34d399; }
    h1 { font-size: 3rem; font-weight: 900; }
    .st-emotion-cache-1gulkj5 { background-color: #1e293b; border-radius: 12px; padding: 20px; }
</style>
""", unsafe_allow_html=True)


# --- GÜVENLİ BAĞLANTI VE ANAHTAR YÖNETİMİ ---
@st.cache_resource
def init_firebase():
    """Firebase bağlantısını güvenli bir şekilde başlatır."""
    try:
        cred_dict = st.secrets["firebase"]
        cred = credentials.Certificate(cred_dict)
    except (KeyError, FileNotFoundError):
        try:
            cred = credentials.Certificate("firebase-key.json")
        except FileNotFoundError:
            return None
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    return True

def get_gemini_api_key():
    """Gemini API anahtarını güvenli bir şekilde alır."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return st.sidebar.text_input("Gemini API Anahtarınızı Girin", type="password", help="Bu sadece yerel testler içindir.")


# --- TÜM ANALİZ VE GRAFİK FONKSİYONLARI ---

def calistir_analiz(df):
    """Tüm finansal metrikleri ve analiz verilerini tek seferde hesaplar."""
    if df.empty: return {"hata": "Veri bulunamadı."}
    try:
        analiz = {}
        analiz['toplam_gelir'] = df['Gelir'].sum()
        analiz['toplam_gider'] = df['Gider'].sum()
        analiz['net_kar'] = analiz['toplam_gelir'] - analiz['toplam_gider']
        
        gider_kategorileri = df[df['Gider'] > 0].groupby('Kategori')['Gider'].sum()
        analiz['en_yuksek_gider_kategorisi'] = gider_kategorileri.idxmax() if not gider_kategorileri.empty else "N/A"
        analiz['kar_marji'] = (analiz['net_kar'] / analiz['toplam_gelir'] * 100) if analiz['toplam_gelir'] > 0 else 0
        
        # Grafik verilerini önceden hesapla
        analiz['aylik_veri'] = df.set_index('Tarih').resample('M').agg({'Gelir': 'sum', 'Gider': 'sum'})
        analiz['aylik_veri']['Net Kar'] = analiz['aylik_veri']['Gelir'] - analiz['aylik_veri']['Gider']
        analiz['aylik_veri']['Kar Marjı'] = (analiz['aylik_veri']['Net Kar'] / analiz['aylik_veri']['Gelir'] * 100).fillna(0)
        
        analiz['top_urunler'] = df[df['Gelir'] > 0].groupby('Satilan_Urun_Adi')['Gelir'].sum().nlargest(5)
        analiz['gider_dagilimi'] = gider_kategorileri
        return analiz
    except Exception as e: return {"hata": str(e)}

def create_gauge_chart(score, title):
    """Finansal Sağlık Skoru için gauge chart oluşturur."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = score,
        title = {'text': title, 'font': {'size': 20}},
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#10b981"},
                 'steps' : [{'range': [0, 40], 'color': '#ef4444'}, {'range': [40, 70], 'color': '#f59e0b'}]}))
    fig.update_layout(paper_bgcolor = "#0f172a", font = {'color': "white"})
    return fig

def prophet_tahmini_yap(aylik_gelir):
    """Prophet modeli ile tahmin yapar."""
    if len(aylik_gelir) < 2: return None, None
    prophet_df = aylik_gelir.reset_index().rename(columns={'Tarih': 'ds', 'Gelir': 'y'})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=3, freq='M')
    forecast = model.predict(future)
    return model, forecast

def yorum_uret(api_key, prompt_data):
    """AI Danışman için kısa yorumlar üretir."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Sen deneyimli bir finansal danışmansın. Şu verilere dayanarak, 1-2 cümlelik kısa ve öz bir yorum yap: {prompt_data}"
        response = model.generate_content(prompt)
        return response.text
    except Exception: return "AI yorumu şu anda kullanılamıyor."

# YENİ: Profesyonel Tahmin Yorumu Üreten Fonksiyon
def tahmin_yorumu_uret(api_key, forecast_df):
    """
    Prophet tahmin sonuçlarını alıp, aktüeryal bir bakış açısıyla profesyonel bir stratejik yorum üretir.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Yorum için gerekli verileri tahminden çıkar
        son_tahmin = forecast_df.iloc[-1]
        onceki_tahmin = forecast_df.iloc[-4]
        trend = "Yükselişte" if son_tahmin['yhat'] > onceki_tahmin['yhat'] else "Düşüşte veya Durgun"
        belirsizlik_araligi = son_tahmin['yhat_upper'] - son_tahmin['yhat_lower']
        
        prompt = f"""
        Sen, aktüerya ve risk yönetimi konusunda uzman, profesyonel bir finansal stratejistsin.
        Aşağıdaki gelecek tahmini verilerini analiz et ve stratejik bir yorum yaz. Yorumun şunları içermeli:
        1. Tahminin ana yönü (trend) hakkında bir değerlendirme.
        2. Tahmindeki belirsizlik aralığına (volatilite) dayalı bir risk analizi.
        3. Bu öngörülere dayanarak şirketin atması gereken 1-2 adet stratejik adım.
        Tonun profesyonel, analitik ve yol gösterici olmalı.

        Veriler:
        - Gelecek 3 Aylık Gelir Tahmini Trendi: {trend}
        - Son Tahmin Edilen Gelir (yhat): {son_tahmin['yhat']:.2f} TL
        - Tahmin Güven Aralığı (En Kötü Senaryo - yhat_lower): {son_tahmin['yhat_lower']:.2f} TL
        - Tahmin Güven Aralığı (En İyi Senaryo - yhat_upper): {son_tahmin['yhat_upper']:.2f} TL
        - Belirsizlik Aralığı Genişliği: {belirsizlik_araligi:.2f} TL (Bu değerin yüksekliği, tahminin daha az kesin olduğunu ve riskin arttığını gösterir.)
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception: 
        return "Stratejik tahmin yorumu şu anda üretilemiyor. Lütfen API anahtarınızı kontrol edin."


# --- ARAYÜZ GÖSTERİM FONKSİYONLARI ---

def show_dashboard(user_info, api_key):
    subscription_plan = user_info.get('subscription_plan', 'None')
    st.title("🚀 Finansal Analiz Paneli")
    st.sidebar.info(f"Aktif Paketiniz: **{subscription_plan}**")
    
    uploaded_file = st.sidebar.file_uploader("CSV dosyanızı yükleyin", type="csv")
    if not uploaded_file:
        st.info("Lütfen analize başlamak için kenar çubuğundan bir CSV dosyası yükleyin.")
        return

    df = pd.read_csv(uploaded_file, parse_dates=['Tarih'])
    analiz = calistir_analiz(df)
    if "hata" in analiz:
        st.error(f"Analiz hatası: {analiz['hata']}"); return

    # --- KRİTİK EŞİK UYARILARI ---
    if analiz['kar_marji'] < 15:
        st.warning(f"⚠️ Kritik Eşik Uyarısı: Kar marjınız (%{analiz['kar_marji']:.2f}) %15'in altında. Maliyetleri gözden geçirin.", icon="🚨")

    # --- SEKMELİ YAPI ---
    tab1, tab2, tab3, tab4 = st.tabs(["Genel Bakış", "Gelir Analizi", "Gider Analizi", "Gelecek Tahmini"])

    with tab1:
        st.header("Genel Finansal Durum")
        skor = max(0, min(100, analiz['kar_marji'] * 2.5))
        st.plotly_chart(create_gauge_chart(skor, "Finansal Sağlık Skoru"), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            fig_bar = px.bar(analiz['aylik_veri'], x=analiz['aylik_veri'].index, y=['Gelir', 'Gider'], title="Aylık Gelir & Gider Karşılaştırması", barmode='group')
            st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            fig_line = px.line(analiz['aylik_veri'], x=analiz['aylik_veri'].index, y='Net Kar', title="Aylık Net Kâr Trendi", markers=True)
            st.plotly_chart(fig_line, use_container_width=True)

    with tab2:
        st.header("Detaylı Gelir Analizi")
        col1, col2 = st.columns(2)
        with col1:
            fig_urun = px.bar(analiz['top_urunler'], x='Gelir', y=analiz['top_urunler'].index, orientation='h', title="En Çok Gelir Getiren 5 Ürün/Hizmet")
            st.plotly_chart(fig_urun, use_container_width=True)
        with col2:
            fig_marj = px.area(analiz['aylik_veri'], x=analiz['aylik_veri'].index, y='Kar Marjı', title="Aylık Kar Marjı (%) Trendi", markers=True)
            st.plotly_chart(fig_marj, use_container_width=True)
        if api_key and not analiz['top_urunler'].empty:
            prompt_data = f"En karlı ürün '{analiz['top_urunler'].index[0]}' ve kar marjı trendi."
            st.info(f"**AI Yorumu:** {yorum_uret(api_key, prompt_data)}")

    with tab3:
        st.header("Detaylı Gider Analizi")
        fig_pie = px.pie(analiz['gider_dagilimi'], names=analiz['gider_dagilimi'].index, values=analiz['gider_dagilimi'].values, title="Kategoriye Göre Gider Dağılımı", hole=.4)
        st.plotly_chart(fig_pie, use_container_width=True)
        if api_key and not analiz['gider_dagilimi'].empty:
            prompt_data = f"En büyük gider kalemi '{analiz['en_yuksek_gider_kategorisi']}'. Bu giderin toplamdaki payı."
            st.info(f"**AI Yorumu:** {yorum_uret(api_key, prompt_data)}")

    with tab4:
        st.header("AI Destekli Gelecek Tahmini")
        aylik_gelir = df.set_index('Tarih')[['Gelir']].resample('M').sum()
        model, tahmin = prophet_tahmini_yap(aylik_gelir)
        if model and tahmin is not None:
            fig_prophet = plot_plotly(model, tahmin, xlabel="Tarih", ylabel="Gelir")
            st.plotly_chart(fig_prophet, use_container_width=True)
            
            # YENİ: Profesyonel AI Yorumu Bölümü
            st.divider()
            st.subheader("🤖 Stratejik Tahmin Analizi")
            if api_key:
                with st.spinner("AI stratejistiniz geleceği yorumluyor..."):
                    stratejik_yorum = tahmin_yorumu_uret(api_key, tahmin)
                    st.markdown(stratejik_yorum)
            else:
                st.warning("Stratejik yorumu görmek için lütfen API anahtarınızı girin.")
        else:
            st.warning("Tahmin oluşturmak için yeterli veri yok.")


def main():
    if 'user_info' not in st.session_state: st.session_state['user_info'] = None
    
    firebase_ok = init_firebase()
    if not firebase_ok:
        st.warning("Firebase bağlantısı kurulamadı.")
        st.stop()
        
    db = firestore.client()

    with st.sidebar:
        st.header("KazKaz AI")
        if st.session_state.get('user_info'):
            st.write(f"Hoş Geldin, {st.session_state['user_info']['email']}")
            if st.button("Çıkış Yap"):
                st.session_state.clear(); st.rerun()
        else:
            st.write("Finansal geleceğinize hoş geldiniz.")

    if st.session_state.get('user_info'):
        user_info = st.session_state['user_info']
        user_doc = db.collection('users').document(user_info['uid']).get()
        user_info['subscription_plan'] = user_doc.to_dict().get('subscription_plan', 'None') if user_doc.exists else 'None'
        
        if user_info['subscription_plan'] == 'None':
            st.title("Abonelik Paketleri")
            if st.button("Pro Paket Seç (₺750/ay)", type="primary"):
                db.collection('users').document(user_info['uid']).set({'subscription_plan': 'Pro'}, merge=True); st.rerun()
        else:
            api_key = get_gemini_api_key()
            show_dashboard(user_info, api_key)
    else:
        # Basitleştirilmiş giriş/kayıt
        st.title("Finansal Analiz Paneline Hoş Geldiniz")
        email = st.text_input("E-posta")
        password = st.text_input("Şifre", type="password")
        if st.button("Giriş Yap", type="primary"):
            try:
                user = auth.get_user_by_email(email)
                st.session_state['user_info'] = {'uid': user.uid, 'email': user.email}; st.rerun()
            except: st.error("E-posta veya şifre hatalı.")

if __name__ == '__main__':
    main()
