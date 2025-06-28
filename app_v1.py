import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import requests
import google.generativeai as genai

# --- FONKSİYONLAR BÖLÜMÜ (Değişiklik yok) ---
# calistir_analiz, doviz_kuru_getir, prophet_tahmini_yap fonksiyonları öncekiyle aynı
def calistir_analiz(veri_df):
    if veri_df.empty: return {"hata": "Filtrelenen veri bulunamadı."}
    try:
        toplam_gelir = veri_df['Gelir'].sum(); toplam_gider = veri_df['Gider'].sum(); net_kar = toplam_gelir - toplam_gider
        gider_kategorileri = veri_df.groupby('Kategori')['Gider'].sum()
        en_yuksek_gider_kategorisi = gider_kategorileri.idxmax() if not gider_kategorileri.empty else "N/A"
        en_yuksek_gider_tutari = gider_kategorileri.max() if not gider_kategorileri.empty else 0
        urun_gelirleri = veri_df.groupby('Satilan_Urun_Adi')['Gelir'].sum()
        en_cok_gelir_getiren_urun = urun_gelirleri.idxmax() if not urun_gelirleri.empty else "N/A"
        return {"toplam_gelir": toplam_gelir, "toplam_gider": toplam_gider, "net_kar": net_kar, "en_yuksek_gider_kategorisi": en_yuksek_gider_kategorisi, "en_yuksek_gider_tutari": en_yuksek_gider_tutari, "en_cok_gelir_getiren_urun": en_cok_gelir_getiren_urun}
    except Exception as e: return {"hata": str(e)}

def doviz_kuru_getir():
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/TRY"); data = response.json()
        usd_kur = 1 / data['rates']['USD']; eur_kur = 1 / data['rates']['EUR']
        return {"USD": usd_kur, "EUR": eur_kur}
    except: return None

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
        Sen, bir şirketin finansal durumunu analiz eden, deneyimli, pozitif ve yol gösterici bir finansal danışmansın. 
        Tonun 'içten, sıcak ve insani' olmalı.
        Analiz edilecek veriler şunlar:
        - Toplam Gelir: {analiz_sonuclari['toplam_gelir']:,} TL
        - Toplam Gider: {analiz_sonuclari['toplam_gider']:,} TL
        - Net Kar: {analiz_sonuclari['net_kar']:,} TL
        - En Büyük Gider Kalemi: {analiz_sonuclari['en_yuksek_gider_kategorisi']} ({analiz_sonuclari['en_yuksek_gider_tutari']:,} TL)
        - Gelecek 3 Aylık Gelir Tahmini Trendi: {tahmin_ozeti}
        Bu verilere dayanarak, şirket için 2-3 paragraftan oluşan bir durum değerlendirmesi ve 2-3 maddelik somut bir eylem planı önerisi yaz. Yorumuna "Değerli Yönetici," diye başla.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Yorum üretilirken bir hata oluştu. API anahtarınızın doğru ve geçerli olduğundan emin olun. Hata: {e}"

# --- ANA UYGULAMA BÖLÜMÜ ---

st.set_page_config(page_title="AI Finans Paneli", layout="wide", initial_sidebar_state="expanded")
st.title("💡 AI Destekli Finansal Danışman")

# DEĞİŞİKLİK: API Anahtarını isteyen kutuyu sildik.
st.sidebar.header("Kontrol Paneli")
uploaded_file = st.sidebar.file_uploader("Analiz için CSV dosyanızı buraya yükleyin", type="csv")

if uploaded_file is not None:
    # Eğer bir dosya yüklendiyse...
    ana_veri = pd.read_csv(uploaded_file, parse_dates=['Tarih'])
    
    # ... (Filtreleme kodları aynı)
    st.sidebar.divider()
    st.sidebar.header("Filtreleme Seçenekleri")
    urun_listesi = ["Tümü"] + sorted(ana_veri['Satilan_Urun_Adi'].unique().tolist())
    secilen_urun = st.sidebar.selectbox("Ürüne Göre Filtrele:", urun_listesi)
    if secilen_urun == "Tümü": filtrelenmis_veri = ana_veri
    else: filtrelenmis_veri = ana_veri[ana_veri['Satilan_Urun_Adi'] == secilen_urun]
    
    # --- Analiz ve Gösterge Paneli ---
    # ... (Tüm metrik, döviz kuru, grafik kodları aynı)
    analiz_sonuclari = calistir_analiz(filtrelenmis_veri)
    if "hata" not in analiz_sonuclari:
        # ... metriklerin gösterimi ...

        # YENİ: AI Danışman Yorumunu, gizli anahtarı kullanarak gösterme
        st.divider()
        st.header("🤖 AI Danışman Yorumu")
        with st.spinner("Yapay zeka danışmanınız verileri analiz ediyor ve size özel bir yorum hazırlıyor..."):
            # DEĞİŞİKLİK: API anahtarını kullanıcıdan değil, Streamlit'in gizli kasasından alıyoruz
            aylik_veri = filtrelenmis_veri.set_index('Tarih')[['Gelir']].resample('ME').sum()
            model, tahmin = prophet_tahmini_yap(aylik_veri)
            if tahmin is not None:
                son_gercek_gelir = tahmin['yhat'].iloc[-4]; son_tahmin_gelir = tahmin['yhat'].iloc[-1]
                tahmin_trendi = "Yükselişte" if son_tahmin_gelir > son_gercek_gelir else "Düşüşte veya Durgun"
                yorum = yorum_uret(st.secrets["GEMINI_API_KEY"], analiz_sonuclari, tahmin_trendi)
                st.markdown(yorum)
            else:
                st.warning("Yorum oluşturmak için yeterli veri bulunmuyor.")
    else:
        st.warning("Veri analizi sırasında bir hata oluştu.")
else:
    # Dosya yüklenmediyse gösterilecek karşılama ekranı
    st.info(" Lütfen analize başlamak için sol taraftaki menüden bir CSV dosyası yükleyin.")