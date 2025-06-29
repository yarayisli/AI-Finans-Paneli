# KazKaz AI Finansal Danisman (Yerel + Bulut Uyumlu Düzeltilmiş Versiyon)
import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import google.generativeai as genai

# --- Sayfa Ayarları ---
st.set_page_config(page_title="KazKaz Finansal Danışman", layout="wide")

# --- CSS Stili ---
st.markdown("""
<style>
    /* ... (CSS Stillerimiz aynı kalıyor) ... */
    .stButton > button { border-radius: 8px; border: 2px solid #10b981; color: #10b981; background-color: transparent; transition: all 0.3s; font-weight: bold; }
    .stButton > button:hover { border-color: #ffffff; color: #ffffff; background-color: #10b981; }
    .st-emotion-cache-1gulkj5 { background-color: #1e293b; border-radius: 12px; padding: 20px; }
    h1, h2, h3 { color: #ffffff; }
</style>
""", unsafe_allow_html=True)


# --- Finansal Analiz Fonksiyonu ---
def calistir_analiz(df):
    try:
        toplam_gelir = df['Gelir'].sum()
        toplam_gider = df['Gider'].sum()
        net_kar = toplam_gelir - toplam_gider
        kategori = df.groupby('Kategori')['Gider'].sum()
        en_yuksek_gider_kategori = kategori.idxmax() if not kategori.empty else "N/A"
        return {
            "toplam_gelir": toplam_gelir,
            "toplam_gider": toplam_gider,
            "net_kar": net_kar,
            "en_yuksek_gider_kategori": en_yuksek_gider_kategori
        }
    except Exception as e:
        return {"hata": str(e)}

# --- AI Yorumu Üreten Fonksiyon ---
def yorum_uret(api_key, analiz, trend):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Sen deneyimli, sıcak kanlı ve yapıcı bir finansal danışmansın.
        Şirket verilerini değerlendirerek 2-3 paragraf durum özeti ve 3 maddelik eylem planı yaz.
        - Toplam Gelir: {analiz['toplam_gelir']:,} TL
        - Net Kar: {analiz['net_kar']:,} TL
        - En Büyük Gider: {analiz['en_yuksek_gider_kategori']}
        - Trend: {trend}
        Yorumuna "Değerli Yönetici," diye başla.
        """
        yanit = model.generate_content(prompt)
        return yanit.text
    except Exception as e:
        # Kullanıcıya daha anlaşılır bir hata mesajı gösterelim
        st.error(f"AI Yorumu üretilirken bir sorun oluştu. API anahtarınızın geçerli olduğundan emin olun.")
        return "" # Hata durumunda boş metin döndür

# --- Gelir Tahmini Fonksiyonu ---
def prophet_tahmini(df):
    try:
        # Prophet'in çalışması için en az 2 veri noktası gerekir
        if len(df) < 2:
            return None, None
        p_df = df.reset_index().rename(columns={"Tarih": "ds", "Gelir": "y"})
        model = Prophet()
        model.fit(p_df)
        future = model.make_future_dataframe(periods=3, freq='M')
        forecast = model.predict(future)
        return model, forecast
    except Exception as e:
        return None, None

# --- Ana Panel ---
def ana_panel():
    st.title("📈 KazKaz AI Destekli Finansal Danışman")
    st.sidebar.title("Kontrol Paneli")

    # DÜZELTİLMİŞ YAPI: Hibrit API Anahtar Yönetimi
    api_key = None
    try:
        # Önce Streamlit Cloud'daki gizli anahtarı dener
        api_key = st.secrets["GEMINI_API_KEY"]
        st.sidebar.success("Cloud API Anahtarı başarıyla yüklendi.")
    except (KeyError, FileNotFoundError):
        # Eğer bulamazsa, yerelde kullanıcıdan ister
        st.sidebar.warning("Cloud anahtarı bulunamadı.")
        api_key = st.sidebar.text_input("Lütfen Gemini API Anahtarınızı Girin:", type="password", help="Bu anahtar sadece yerel testler için gereklidir.")

    st.sidebar.divider()
    
    # Dosya yükleyiciyi kenar çubuğuna taşıdık
    dosya = st.sidebar.file_uploader("Lütfen CSV dosyanızı yükleyin", type="csv")
    
    if not dosya:
        st.info("Devam etmek için lütfen kenar çubuğundan bir CSV dosyası yükleyin.")
        return

    df = pd.read_csv(dosya, parse_dates=['Tarih'])
    analiz = calistir_analiz(df)

    if "hata" in analiz:
        st.error(f"Veri analizi sırasında hata: {analiz['hata']}")
        return

    # Metrikleri ana panelde göster
    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Gelir", f"{analiz['toplam_gelir']:,} TL")
    col2.metric("Toplam Gider", f"{analiz['toplam_gider']:,} TL")
    col3.metric("Net Kar", f"{analiz['net_kar']:,} TL")

    st.divider()
    st.header("🔮 Gelecek Gelir Tahmini")
    aylik = df.set_index('Tarih')[['Gelir']].resample('M').sum()
    model, forecast = prophet_tahmini(aylik)
    
    trend = "Yetersiz veri"
    if model and forecast is not None:
        fig = plot_plotly(model, forecast, xlabel="Tarih", ylabel="Gelir")
        st.plotly_chart(fig, use_container_width=True)
        trend = "Yükselişte" if forecast['yhat'].iloc[-1] > forecast['yhat'].iloc[-4] else "Düşüşte/Durgun"
    else:
        st.warning("Tahmin oluşturmak için yeterli veri yok (en az 2 aylık veri gerekir).")

    st.divider()
    st.header("🤖 AI Danışman Yorumu")
    
    # DÜZELTİLMİŞ YAPI: API anahtarı varsa yorumu üret
    if api_key:
        with st.spinner("AI yorum üretiyor..."):
            yorum = yorum_uret(api_key, analiz, trend)
            st.markdown(yorum)
    else:
        st.warning("AI yorumunu görmek için lütfen kenar çubuğundan geçerli bir API anahtarı girin.")


if __name__ == '__main__':
    ana_panel()
