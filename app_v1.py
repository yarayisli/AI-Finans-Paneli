import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import requests

# --- FONKSİYONLAR BÖLÜMÜ (Değişiklik yok) ---
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

# --- ANA UYGULAMA BÖLÜMÜ ---

st.set_page_config(page_title="AI Finans Paneli", layout="wide", initial_sidebar_state="expanded")
st.title("🚀 AI Destekli Finansal Analiz Paneli v2.1")

# DEĞİŞİKLİK: Kenar çubuğunu en başa taşıdık ve dosya yükleyici ekledik
st.sidebar.header("Kontrol Paneli")

# YENİ: Dosya Yükleyici
uploaded_file = st.sidebar.file_uploader("Analiz için CSV dosyanızı buraya yükleyin", type="csv")

# DEĞİŞİKLİK: Tüm uygulama artık bir dosya yüklenip yüklenmediğine bağlı çalışacak
if uploaded_file is not None:
    # Eğer bir dosya yüklendiyse...
    ana_veri = pd.read_csv(uploaded_file, parse_dates=['Tarih'])
    
    st.sidebar.divider()
    st.sidebar.header("Filtreleme Seçenekleri")
    urun_listesi = ["Tümü"] + sorted(ana_veri['Satilan_Urun_Adi'].unique().tolist())
    secilen_urun = st.sidebar.selectbox("Ürüne Göre Filtrele:", urun_listesi)

    if secilen_urun == "Tümü":
        filtrelenmis_veri = ana_veri
    else:
        filtrelenmis_veri = ana_veri[ana_veri['Satilan_Urun_Adi'] == secilen_urun]
    
    # --- Analiz ve Gösterge Paneli ---
    kurlar = doviz_kuru_getir()
    if kurlar:
        st.subheader("Anlık Kurlar")
        col_kur1, col_kur2 = st.columns(2)
        col_kur1.metric("USD/TRY", f"{kurlar['USD']:.2f} ₺")
        col_kur2.metric("EUR/TRY", f"{kurlar['EUR']:.2f} ₺")
    st.divider()

    analiz_sonuclari = calistir_analiz(filtrelenmis_veri)
    if "hata" not in analiz_sonuclari:
        st.header(f"'{secilen_urun}' için Finansal Durum")
        col1, col2, col3 = st.columns(3)
        col1.metric("Toplam Gelir", f"{analiz_sonuclari['toplam_gelir']} TL")
        col2.metric("Toplam Gider", f"{analiz_sonuclari['toplam_gider']} TL")
        col3.metric("Net Kar", f"{analiz_sonuclari['net_kar']} TL")
        
        st.divider()
        st.header(f"'{secilen_urun}' için Profesyonel Gelir Tahmini")
        aylik_veri = filtrelenmis_veri.set_index('Tarih')[['Gelir']].resample('ME').sum()
        
        model, tahmin = prophet_tahmini_yap(aylik_veri)
        if model and tahmin is not None:
            fig = plot_plotly(model, tahmin, xlabel="Tarih", ylabel="Gelir")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Tahmin oluşturmak için yeterli veri bulunmuyor (en az 2 ay gerekir).")
    else:
        st.warning(f"'{secilen_urun}' için gösterilecek veri bulunmuyor.")

else:
    # YENİ: Eğer henüz dosya yüklenmediyse gösterilecek karşılama ekranı
    st.info(" Lütfen analize başlamak için sol taraftaki menüden bir CSV dosyası yükleyin.")
    st.subheader("CSV Dosyanız Nasıl Olmalı?")
    st.markdown("""
    Yükleyeceğiniz CSV dosyası, analizlerin doğru çalışması için aşağıdaki sütun başlıklarını içermelidir:
    - `Tarih` (YYYY-AA-GG formatında, örn: 2025-01-15)
    - `Gelir` (Sayısal değer)
    - `Gider` (Sayısal değer)
    - `Kategori` (Giderin kategorisi, örn: Maaş, Kira)
    - `Satilan_Urun_Adi` (Satılan ürünün veya hizmetin adı)
    
    Örnek olarak projemizde kullandığımız `ornek_veri.csv` dosyasını kullanabilirsiniz.
    """)