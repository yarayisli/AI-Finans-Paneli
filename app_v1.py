# KazKaz AI Finansal Danisman (Yerel + Bulut Uyumlu Versiyon)
import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import google.generativeai as genai

# --- Sayfa Ayarları ---
st.set_page_config(page_title="KazKaz Finansal Danışman", layout="wide")

# --- Secret Anahtarı Hibrit Sistemi ---
def get_gemini_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY")

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
        return f"AI Yorum Hatası: {e}"

# --- Gelir Tahmini ---
def prophet_tahmini(df):
    try:
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

    dosya = st.file_uploader("Lütfen CSV dosyanızı yükleyin", type="csv")
    if not dosya:
        st.info("Devam etmek için bir CSV dosyası yükleyin. Örnek şablon isterseniz iletişime geçin.")
        return

    df = pd.read_csv(dosya, parse_dates=['Tarih'])
    analiz = calistir_analiz(df)

    if "hata" in analiz:
        st.error(f"Veri analizi sırasında hata: {analiz['hata']}")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Gelir", f"{analiz['toplam_gelir']:,} TL")
    col2.metric("Toplam Gider", f"{analiz['toplam_gider']:,} TL")
    col3.metric("Net Kar", f"{analiz['net_kar']:,} TL")

    st.divider()
    st.header("🔮 Gelecek Gelir Tahmini")
    aylik = df.set_index('Tarih')[['Gelir']].resample('M').sum()
    model, forecast = prophet_tahmini(aylik)
    if model and forecast is not None:
        st.plotly_chart(plot_plotly(model, forecast))
        trend = "Yükselişte" if forecast['yhat'].iloc[-1] > forecast['yhat'].iloc[-4] else "Düşüşte/Durgun"
    else:
        trend = "Yetersiz veri"
        st.warning("Tahmin oluşturmak için yeterli veri yok.")

    st.divider()
    st.header("🤖 AI Danışman Yorum")
    with st.spinner("AI yorum üretiyor..."):
        api_key = get_gemini_api_key()
        yorum = yorum_uret(api_key, analiz, trend)
        st.markdown(yorum)

if __name__ == '__main__':
    ana_panel()
