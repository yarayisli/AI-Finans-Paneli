# KazKaz AI Finansal Danışman - Geliştirilmiş Son Sürüm
import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import google.generativeai as genai
import os # Ortam değişkenleri için eklendi

# --- Sayfa Yapılandırması ve Stil ---
st.set_page_config(page_title="KazKaz Finansal Danışman", layout="wide", initial_sidebar_state="expanded")

# --- CSS Stili (Profesyonel, Mobil Uyumluluk ve Renk Paleti Dahil) ---
st.markdown("""
<style>
    /* Ana Gövde ve Fontlar */
    body {
        background-color: #0f172a;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    .block-container { 
        padding-top: 2rem; padding-bottom: 2rem; padding-left: 2rem; padding-right: 2rem; 
    }
    .st-emotion-cache-16txtl3 { 
        background-color: #0f172a; 
    }
    .stButton > button {
        border-radius: 8px; border: 2px solid #10b981; color: white; background-color: #10b981;
        transition: all 0.3s; font-weight: bold; padding: 10px 24px; width: 100%;
    }
    .stButton > button:hover {
        border-color: #34d399; color: white; background-color: #34d399;
    }
    .st-emotion-cache-1gulkj5 { 
        background-color: #1e293b !important; 
        border: 1px solid #334155; 
        border-radius: 12px; 
        padding: 20px;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .stTextInput > div > div > input { 
        background-color: #1e293b; 
        color: white; 
        border-radius: 8px; 
    }
</style>
""", unsafe_allow_html=True)


# --- Fonksiyonlar ---

def calistir_analiz(df):
    """Veriyi analiz eder ve temel finansal metrikleri döndürür."""
    try:
        toplam_gelir = df['Gelir'].sum()
        toplam_gider = df['Gider'].sum()
        net_kar = toplam_gelir - toplam_gider
        kategori = df.groupby('Kategori')['Gider'].sum()
        en_yuksek_gider_kategori = kategori.idxmax() if not kategori.empty else "N/A"
        kar_marji = (net_kar / toplam_gelir * 100) if toplam_gelir > 0 else 0
        
        return {
            "toplam_gelir": toplam_gelir,
            "toplam_gider": toplam_gider,
            "net_kar": net_kar,
            "en_yuksek_gider_kategorisi": en_yuksek_gider_kategori,
            "kar_marji": kar_marji
        }
    except Exception as e:
        return {"hata": str(e)}

def prophet_tahmini(df):
    """Prophet modeli ile gelir tahmini yapar."""
    try:
        if len(df) < 2:
            return None, None
        p_df = df.reset_index().rename(columns={"Tarih": "ds", "Gelir": "y"})
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(p_df)
        future = model.make_future_dataframe(periods=3, freq='M')
        forecast = model.predict(future)
        return model, forecast
    except Exception:
        return None, None

def yorum_uret(api_key, analiz, trend):
    """AI Danışman için yorum üretir."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Sen deneyimli, sıcak kanlı ve yapıcı bir finansal danışmansın.
        Aşağıdaki şirket verilerini değerlendirerek 2-3 paragrafta mevcut durumu özetle ve 3 maddelik bir eylem planı önerisi yaz.
        - Toplam Gelir: {analiz['toplam_gelir']:,} TL
        - Net Kar: {analiz['net_kar']:,} TL
        - Kar Marjı: %{analiz['kar_marji']:.2f}
        - En Büyük Gider Kalemi: {analiz['en_yuksek_gider_kategorisi']}
        - Gelecek Tahmini Trendi: {trend}
        Yorumuna "Değerli Yönetici," diye başla.
        """
        yanit = model.generate_content(prompt)
        return yanit.text
    except Exception as e:
        return f"AI Yorumu üretilirken bir hata oluştu. Lütfen API anahtarınızı kontrol edin. Hata: {e}"

# --- Ana Panel ---

def ana_panel():
    st.title("📈 KazKaz AI Destekli Finansal Danışman")
    st.sidebar.title("Kontrol Paneli")

    # DÜZELTİLMİŞ YAPI: Hibrit API Anahtar Yönetimi
    api_key = None
    try:
        # Önce Streamlit Cloud'daki gizli anahtarı dener
        api_key = st.secrets["GEMINI_API_KEY"]
        st.sidebar.success("Cloud API Anahtarı başarıyla yüklendi.", icon="☁️")
    except (KeyError, FileNotFoundError):
        # Eğer bulamazsa, yerelde kullanıcıdan ister
        st.sidebar.warning("Cloud anahtarı bulunamadı. Lütfen yerel anahtarınızı girin.")
        api_key = st.sidebar.text_input("Gemini API Anahtarınızı Girin:", type="password", help="Bu anahtar sadece yerel testler için gereklidir.")

    st.sidebar.divider()
    
    dosya = st.sidebar.file_uploader("Lütfen CSV dosyanızı yükleyin", type="csv")
    
    if not dosya:
        st.info("Devam etmek için lütfen kenar çubuğundan bir CSV dosyası yükleyin.")
        st.markdown("---")
        st.subheader("Örnek CSV Şablonu")
        st.markdown("Yükleyeceğiniz dosyanın aşağıdaki sütun başlıklarını içermesi gerekmektedir:")
        st.code("Tarih,Gelir,Gider,Kategori,Satilan_Urun_Adi")
        return

    df = pd.read_csv(dosya, parse_dates=['Tarih'])
    analiz = calistir_analiz(df)

    if "hata" in analiz:
        st.error(f"Veri analizi sırasında hata: {analiz['hata']}")
        return

    # Metrikleri ana panelde göster
    st.subheader("Finansal Özet")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Toplam Gelir", f"{analiz['toplam_gelir']:,} TL")
    col2.metric("Toplam Gider", f"{analiz['toplam_gider']:,} TL")
    col3.metric("Net Kar", f"{analiz['net_kar']:,} TL")
    col4.metric("Kar Marjı", f"%{analiz.get('kar_marji', 0):.2f}")

    st.divider()

    col_grafik, col_yorum = st.columns([2, 1])

    with col_grafik:
        st.subheader("🔮 Gelecek Gelir Tahmini")
        aylik = df.set_index('Tarih')[['Gelir']].resample('M').sum()
        model, forecast = prophet_tahmini(aylik)
        
        trend = "Yetersiz veri"
        if model and forecast is not None:
            fig = plot_plotly(model, forecast, xlabel="Tarih", ylabel="Gelir")
            st.plotly_chart(fig, use_container_width=True)
            trend = "Yükselişte" if forecast['yhat'].iloc[-1] > forecast['yhat'].iloc[-4] else "Düşüşte/Durgun"
        else:
            st.warning("Tahmin oluşturmak için yeterli veri yok (en az 2 aylık veri gerekir).")

    with col_yorum:
        st.subheader("🤖 AI Danışman Yorumu")
        if api_key:
            with st.spinner("AI yorum üretiyor..."):
                yorum = yorum_uret(api_key, analiz, trend)
                st.markdown(yorum)
        else:
            st.warning("AI yorumunu görmek için lütfen kenar çubuğundan geçerli bir API anahtarı girin.")

if __name__ == '__main__':
    ana_panel()
