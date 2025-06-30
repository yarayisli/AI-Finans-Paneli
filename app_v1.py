# KazKaz AI Finansal Danışman - Gelişmiş Analiz Paneli
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
    try:
        cred = credentials.Certificate(st.secrets["firebase"])
    except (KeyError, FileNotFoundError):
        try:
            cred = credentials.Certificate("firebase-key.json")
        except FileNotFoundError:
            return None
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    return True

def get_gemini_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return st.sidebar.text_input("Gemini API Anahtarınızı Girin", type="password", help="Bu sadece yerel testler içindir.")


# --- TÜM ANALİZ VE GRAFİK FONKSİYONLARI ---

def calistir_analiz(df):
    """Tüm finansal metrikleri hesaplar."""
    if df.empty: return {"hata": "Veri bulunamadı."}
    try:
        analiz = {}
        analiz['toplam_gelir'] = df['Gelir'].sum()
        analiz['toplam_gider'] = df['Gider'].sum()
        analiz['net_kar'] = analiz['toplam_gelir'] - analiz['toplam_gider']
        gider_kategorileri = df[df['Gider'] > 0].groupby('Kategori')['Gider'].sum()
        analiz['en_yuksek_gider_kategorisi'] = gider_kategorileri.idxmax() if not gider_kategorileri.empty else "N/A"
        analiz['kar_marji'] = (analiz['net_kar'] / analiz['toplam_gelir'] * 100) if analiz['toplam_gelir'] > 0 else 0
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
        mode = "gauge+number",
        value = score,
        title = {'text': title, 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#10b981"},
            'bgcolor': "#1e293b",
            'borderwidth': 2,
            'bordercolor': "#334155",
            'steps' : [
                {'range': [0, 40], 'color': '#ef4444'},
                {'range': [40, 70], 'color': '#f59e0b'}],
            'threshold' : {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 70}}))
    fig.update_layout(paper_bgcolor = "#0f172a", font = {'color': "white", 'family': "Segoe UI"})
    return fig

def create_bar_chart(df, x, y_list, title):
    """Gelir & Gider gibi karşılaştırmalı çubuk grafik oluşturur."""
    fig = px.bar(df, x=x, y=y_list, title=title, barmode='group')
    fig.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", font_color="white", legend_title_text='')
    return fig

def create_line_chart(df, x, y, title):
    """Net Kar gibi trend çizgisi grafiği oluşturur."""
    fig = px.line(df, x=x, y=y, title=title, markers=True)
    fig.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", font_color="white")
    return fig
    
def create_pie_chart(df, names, values, title):
    """Gider dağılımı gibi pasta grafik oluşturur."""
    fig = px.pie(df, names=names, values=values, title=title, hole=.4)
    fig.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", font_color="white")
    return fig

def yorum_uret(api_key, prompt_data):
    """AI Danışman yorumu üretir."""
    try:
        genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Sen deneyimli bir finansal danışmansın. Şu verilere dayanarak, 1-2 cümlelik kısa ve öz bir yorum yap: {prompt_data}"
        response = model.generate_content(prompt); return response.text
    except Exception: return "AI yorumu şu anda kullanılamıyor."


# --- ARAYÜZ GÖSTERİM FONKSİYONLARI ---

def show_dashboard(user_info, api_key):
    st.title(f"🚀 Finansal Analiz Paneli")
    st.sidebar.info(f"Aktif Paketiniz: **{user_info['subscription_plan']}**")
    
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

    with tab1: # GENEL BAKIŞ
        st.header("Genel Finansal Durum")
        # 1. Finansal Sağlık Skoru
        skor = max(0, min(100, analiz['kar_marji'] * 2.5)) # Basit bir skor hesaplama
        st.plotly_chart(create_gauge_chart(skor, "Finansal Sağlık Skoru"), use_container_width=True)
        
        # 2. Gelir & Gider ve Net Kar Trendleri
        col1, col2 = st.columns(2)
        with col1:
            fig_gelir_gider = create_bar_chart(analiz['aylik_veri'], analiz['aylik_veri'].index, ['Gelir', 'Gider'], "Aylık Gelir & Gider Karşılaştırması")
            st.plotly_chart(fig_gelir_gider, use_container_width=True)
        with col2:
            fig_net_kar = create_line_chart(analiz['aylik_veri'], analiz['aylik_veri'].index, 'Net Kar', "Aylık Net Kâr Trendi")
            st.plotly_chart(fig_net_kar, use_container_width=True)

    with tab2: # GELİR ANALİZİ
        st.header("Detaylı Gelir Analizi")
        # 3. Ürün/Müşteri Bazlı Gelir Grafiği
        fig_top_urunler = px.bar(analiz['top_urunler'], x='Gelir', y=analiz['top_urunler'].index, orientation='h', title="En Çok Gelir Getiren 5 Ürün/Hizmet")
        fig_top_urunler.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", font_color="white", yaxis_title='')
        st.plotly_chart(fig_top_urunler, use_container_width=True)
        
        # BONUS: AI Yorumu
        if api_key and not analiz['top_urunler'].empty:
            prompt_data = f"En çok gelir getiren ürün '{analiz['top_urunler'].index[0]}' ve geliri {int(analiz['top_urunler'].iloc[0])} TL."
            yorum = yorum_uret(api_key, prompt_data)
            st.info(f"**AI Yorumu:** {yorum}")

    with tab3: # GİDER ANALİZİ
        st.header("Detaylı Gider Analizi")
        col1, col2 = st.columns(2)
        with col1:
            # 4. Gider Dağılımı Pastası
            fig_gider_pie = create_pie_chart(analiz['gider_dagilimi'], analiz['gider_dagilimi'].index, analiz['gider_dagilimi'].values, "Kategoriye Göre Gider Dağılımı")
            st.plotly_chart(fig_gider_pie, use_container_width=True)
        with col2:
            # 5. Kar Marjı Zaman Serisi
            fig_kar_marji = create_line_chart(analiz['aylik_veri'], analiz['aylik_veri'].index, 'Kar Marjı', "Aylık Kar Marjı (%) Trendi")
            st.plotly_chart(fig_kar_marji, use_container_width=True)

    with tab4: # GELECEK TAHMİNİ
        st.header("AI Destekli Gelecek Tahmini")
        # 6. Prophet Tahmini
        aylik_gelir = df.set_index('Tarih')[['Gelir']].resample('M').sum()
        model, tahmin = prophet_tahmini_yap(aylik_gelir)
        if model and tahmin is not None:
            fig_prophet = plot_plotly(model, tahmin, xlabel="Tarih", ylabel="Gelir")
            st.plotly_chart(fig_prophet, use_container_width=True)
        else:
            st.warning("Tahmin oluşturmak için yeterli veri yok.")

# --- ANA UYGULAMA AKIŞI ---
def main():
    if not init_firebase(): st.stop()
    if 'user_info' not in st.session_state: st.session_state['user_info'] = None
    if 'page' not in st.session_state: st.session_state['page'] = 'landing'
    # ... (Geri kalan tüm giriş/kayıt ve sayfa yönlendirme mantığı aynı)

if __name__ == '__main__':
    # Bu kısmı basitleştirilmiş bir akışla yeniden yazıyoruz
    if 'user_info' not in st.session_state: st.session_state['user_info'] = None
    if not init_firebase(): 
        st.error("Uygulama başlatılamıyor. Firebase yapılandırması eksik.")
        st.stop()
        
    db = firestore.client()
    
    with st.sidebar:
        st.header("KazKaz AI")
        if st.session_state.get('user_info'):
            st.write(f"Hoş Geldin, {st.session_state['user_info']['email']}")
            if st.button("Çıkış Yap"):
                st.session_state.clear()
                st.rerun()
        else:
            st.write("Finansal geleceğinize hoş geldiniz.")

    if st.session_state.get('user_info'):
        api_key = get_gemini_api_key()
        show_dashboard(st.session_state['user_info'], api_key)
    else:
        # Basitleştirilmiş giriş ekranı
        st.title("Finansal Analiz Paneline Hoş Geldiniz")
        email = st.text_input("E-posta")
        password = st.text_input("Şifre", type="password")
        if st.button("Giriş Yap", type="primary"):
            try:
                user = auth.get_user_by_email(email)
                st.session_state['user_info'] = {'uid': user.uid, 'email': user.email}
                st.rerun()
            except:
                st.error("E-posta veya şifre hatalı.")
