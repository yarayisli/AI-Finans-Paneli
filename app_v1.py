# KazKaz AI Finansal Danışman v2.0 - Gelişmiş ve Katmanlı Yetenekler
import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, auth, firestore
from prophet import Prophet
from prophet.plot import plot_plotly
import google.generativeai as genai
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import io
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os # YENİ: Font yolu için os modülü eklendi

# --- Sayfa Yapılandırması ve Stil ---
st.set_page_config(page_title="KazKaz Finansal Danışman", layout="wide", initial_sidebar_state="auto")
st.markdown("""
<style>
    /* ... (CSS Stillerimiz aynı kalıyor ve geliştiriliyor) ... */
    .stButton > button { border-radius: 8px; border: 2px solid #10b981; color: white; background-color: #10b981; transition: all 0.3s; font-weight: bold; padding: 10px 24px; width: 100%; }
    .stButton > button:hover { border-color: #34d399; color: white; background-color: #34d399; }
    h1 { font-size: 2.8rem; font-weight: 900; }
    .st-emotion-cache-1gulkj5 { background-color: #1e293b; border-radius: 12px; padding: 20px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #0f172a; border-radius: 8px; }
    .stTabs [aria-selected="true"] { background-color: #10b981; }
</style>
""", unsafe_allow_html=True)


# --- GÜVENLİ BAĞLANTI VE ANAHTAR YÖNETİMİ ---
# ... (Bu bölümde değişiklik yok) ...
@st.cache_resource
def init_firebase():
    """Firebase bağlantısını güvenli bir şekilde başlatır."""
    try:
        cred_dict = st.secrets["firebase"]
        cred = credentials.Certificate(cred_dict)
    except Exception:
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
    except KeyError:
        return st.sidebar.text_input("Gemini API Anahtarınızı Girin", type="password", help="Bu sadece yerel testler içindir.")

@st.cache_resource
def init_gspread():
    """Google Sheets API bağlantısını başlatır."""
    try:
        creds_json = st.secrets["gcp_service_account"]
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, scope)
        client = gspread.authorize(creds)
        return client
    except Exception:
        st.sidebar.error("Google Sheets bağlantısı için 'gcp_service_account' secret'ı bulunamadı.")
        return None

# --- VERİ YÜKLEME VE DOĞRULAMA FONKSİYONLARI ---
# ... (Bu bölümde değişiklik yok) ...
def load_from_gsheets(client, url):
    """Google Sheets URL'sinden veri yükler ve DataFrame'e çevirir."""
    try:
        sheet = client.open_by_url(url).sheet1
        data = sheet.get_all_records()
        return pd.DataFrame(data)
    except gspread.exceptions.SpreadsheetNotFound:
        return "Hata: Google Sheet bulunamadı. URL'yi veya paylaşım ayarlarını kontrol edin."
    except Exception as e:
        return f"Hata: Veri okunurken bir sorun oluştu: {str(e)}"

def validate_and_load_data(source, input_data):
    """Veriyi yükler, doğrular ve hataları yönetir."""
    df = None
    try:
        if source == "Dosya Yükle":
            if input_data.name.endswith('.csv'):
                df = pd.read_csv(input_data)
            elif input_data.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(input_data)
        elif source == "Google Sheets":
            if isinstance(input_data, pd.DataFrame):
                df = input_data
            else: # Hata mesajı geldi
                st.error(input_data)
                return None, input_data

        required_columns = ['Tarih', 'Gelir', 'Gider']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            error_msg = f"Hata: Yüklenen veride şu sütunlar eksik: {', '.join(missing_cols)}"
            st.error(error_msg)
            return None, error_msg

        df['Tarih'] = pd.to_datetime(df['Tarih'], errors='coerce')
        if df['Tarih'].isnull().any():
            error_msg = "Hata: 'Tarih' sütunundaki bazı değerler anlaşılamadı. Lütfen 'YYYY-MM-DD' formatını kullanın."
            st.error(error_msg)
            return None, error_msg

        return df, None

    except Exception as e:
        error_msg = f"Veri işlenirken beklenmedik bir hata oluştu: {e}"
        st.error(error_msg)
        return None, error_msg

# --- TÜM ANALİZ VE GRAFİK FONKSİYONLARI ---
# ... (calistir_analiz, create_gauge_chart, prophet_tahmini_yap, yorum_uret, tahmin_yorumu_uret fonksiyonlarında değişiklik yok) ...
def calistir_analiz(df):
    """Tüm finansal metrikleri ve analiz verilerini tek seferde hesaplar."""
    if df.empty: return {"hata": "Veri bulunamadı."}
    try:
        analiz = {}
        df['Gelir'] = pd.to_numeric(df['Gelir'], errors='coerce').fillna(0)
        df['Gider'] = pd.to_numeric(df['Gider'], errors='coerce').fillna(0)
        
        analiz['toplam_gelir'] = df['Gelir'].sum()
        analiz['toplam_gider'] = df['Gider'].sum()
        analiz['net_kar'] = analiz['toplam_gelir'] - analiz['toplam_gider']
        
        gider_kategorileri = df[df['Gider'] > 0].groupby('Kategori')['Gider'].sum()
        analiz['en_yuksek_gider_kategorisi'] = gider_kategorileri.idxmax() if not gider_kategorileri.empty else "N/A"
        analiz['kar_marji'] = (analiz['net_kar'] / analiz['toplam_gelir'] * 100) if analiz['toplam_gelir'] > 0 else 0
        
        analiz['aylik_veri'] = df.set_index('Tarih').resample('M').agg({'Gelir': 'sum', 'Gider': 'sum'})
        analiz['aylik_veri']['Net Kar'] = analiz['aylik_veri']['Gelir'] - analiz['aylik_veri']['Gider']
        analiz['aylik_veri']['Kar Marjı'] = (analiz['aylik_veri']['Net Kar'] / analiz['aylik_veri']['Gelir'] * 100).fillna(0)
        
        analiz['top_urunler'] = df[df['Gelir'] > 0].groupby('Satilan_Urun_Adi')['Gelir'].sum().nlargest(5) if 'Satilan_Urun_Adi' in df.columns else pd.Series()
        analiz['gider_dagilimi'] = gider_kategorileri
        
        analiz['fig_bar'] = px.bar(analiz['aylik_veri'], x=analiz['aylik_veri'].index, y=['Gelir', 'Gider'], title="Aylık Gelir & Gider", barmode='group')
        analiz['fig_line'] = px.line(analiz['aylik_veri'], x=analiz['aylik_veri'].index, y='Net Kar', title="Aylık Net Kâr Trendi", markers=True)
        if not analiz['top_urunler'].empty:
            analiz['fig_urun'] = px.bar(analiz['top_urunler'], x='Gelir', y=analiz['top_urunler'].index, orientation='h', title="En Çok Gelir Getirenler")
        else:
            analiz['fig_urun'] = None
        analiz['fig_marj'] = px.area(analiz['aylik_veri'], x=analiz['aylik_veri'].index, y='Kar Marjı', title="Aylık Kar Marjı (%) Trendi", markers=True)
        if not analiz['gider_dagilimi'].empty:
            analiz['fig_pie'] = px.pie(analiz['gider_dagilimi'], names=analiz['gider_dagilimi'].index, values=analiz['gider_dagilimi'].values, title="Gider Dağılımı", hole=.4)
        else:
            analiz['fig_pie'] = None
            
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

def tahmin_yorumu_uret(api_key, forecast_df):
    """Prophet tahmin sonuçlarını alıp, aktüeryal bir bakış açısıyla profesyonel bir stratejik yorum üretir."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
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
    except Exception: return "Stratejik tahmin yorumu şu anda üretilemiyor. Lütfen API anahtarınızı kontrol edin."


# --- PDF BÖLÜMÜ GÜNCELLEMESİ ---
class PDF(FPDF):
    def header(self):
        # GÜNCELLENDİ: Unicode fontu kullan
        self.set_font('DejaVu', 'B', 15)
        self.cell(0, 10, 'KazKaz AI Finansal Analiz Raporu', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        # GÜNCELLENDİ: Unicode fontu kullan
        self.set_font('DejaVu', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        # GÜNCELLENDİ: Unicode fontu kullan
        self.set_font('DejaVu', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()
        
    def add_metric(self, label, value):
        # GÜNCELLENDİ: Unicode fontu kullan
        self.set_font('DejaVu', 'B', 10)
        self.cell(95, 8, label, 1, 0, 'L')
        self.set_font('DejaVu', '', 10)
        # GÜNCELLENDİ: Hataları önlemek için değeri string'e çevir
        self.cell(95, 8, str(value), 1, 1, 'R')

def generate_pdf_report(analiz, stratejik_yorum=None, forecast_fig=None):
    pdf = PDF()
    
    # YENİ: Unicode fontunu FPDF'e ekle
    # Bu satırın çalışması için DejaVuSans.ttf dosyasının
    # projenizin ana klasöründe olması gerekir.
    try:
        font_path = os.path.join(os.path.dirname(__file__), "DejaVuSans.ttf")
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.add_font("DejaVu", "B", font_path, uni=True) # Kalın (Bold) versiyonu için
    except FileNotFoundError:
        st.error("PDF Raporu için 'DejaVuSans.ttf' font dosyası bulunamadı. Lütfen fontu proje klasörüne ekleyin.")
        return None # Font yoksa PDF oluşturmayı durdur
        
    pdf.add_page()

    # Genel Bakış
    pdf.chapter_title('Genel Finansal Durum')
    pdf.add_metric('Toplam Gelir:', f"{analiz['toplam_gelir']:,.2f} TL")
    pdf.add_metric('Toplam Gider:', f"{analiz['toplam_gider']:,.2f} TL")
    pdf.add_metric('Net Kar:', f"{analiz['net_kar']:,.2f} TL")
    pdf.add_metric('Kar Marjı:', f"{analiz['kar_marji']:.2f}%")
    pdf.add_metric('En Yüksek Gider Kategorisi:', analiz['en_yuksek_gider_kategorisi'])
    pdf.ln(10)

    # Grafikleri resim olarak kaydet ve ekle
    image_files = []
    try:
        for name, fig in analiz.items():
            if name.startswith("fig_") and fig:
                filename = f"temp_{name}.png"
                fig.write_image(filename, scale=2)
                image_files.append(filename)
                
                if name == "fig_bar": pdf.chapter_title('Aylık Gelir & Gider')
                if name == "fig_line": pdf.chapter_title('Aylık Net Kar Trendi')
                if name == "fig_pie": pdf.chapter_title('Gider Dağılımı')
                
                pdf.image(filename, x=None, y=None, w=180)
                pdf.ln(5)

        # Tahmin Grafiği
        if forecast_fig:
            filename = "temp_forecast.png"
            forecast_fig.write_image(filename, scale=2)
            image_files.append(filename)
            pdf.chapter_title('Gelecek Gelir Tahmini')
            pdf.image(filename, x=None, y=None, w=180)
            pdf.ln(5)

        # Stratejik Yorum
        if stratejik_yorum:
            pdf.chapter_title('Stratejik Tahmin Analizi')
            pdf.chapter_body(stratejik_yorum)

    finally:
        # Geçici resim dosyalarını sil
        for f in image_files:
            if os.path.exists(f):
                os.remove(f)
    
    # GÜNCELLENDİ: PDF çıktısını hatasız almak için 'latin-1' yerine 'utf-8' denenebilir
    # Ancak FPDF'in standart çıktısı genellikle 'latin-1' ile uyumludur.
    return bytes(pdf.output(dest='S'))


# --- Geri kalan kodda değişiklik yok ---
# YENİ: Geri Bildirim Kaydetme Fonksiyonu
def log_feedback(db, user_id, feedback_value, yorum):
    """Kullanıcı geri bildirimini Firestore'a kaydeder."""
    feedback_ref = db.collection('feedback').document()
    feedback_ref.set({
        'user_id': user_id,
        'feedback': feedback_value,
        'yorum': yorum,
        'timestamp': firestore.SERVER_TIMESTAMP
    })
    st.toast(f"Geri bildiriminiz için teşekkürler!", icon="✅")

# --- ARAYÜZ GÖSTERİM FONKSİYONLARI ---

def show_dashboard(user_info, api_key, db):
    subscription_plan = user_info.get('subscription_plan', 'Temel')
    st.sidebar.success(f"Aktif Paketiniz: **{subscription_plan}**")

    st.sidebar.header("1. Veri Kaynağınızı Seçin")
    data_source_option = st.sidebar.selectbox("Veri Kaynağı", ["Dosya Yükle", "Google Sheets ile Bağlan"])
    
    df = None
    input_data = None

    if data_source_option == "Dosya Yükle":
        input_data = st.sidebar.file_uploader("CSV veya Excel dosyanızı yükleyin", type=["csv", "xlsx", "xls"])
        if input_data:
            df, error = validate_and_load_data("Dosya Yükle", input_data)
    elif data_source_option == "Google Sheets ile Bağlan":
        gspread_client = init_gspread()
        if gspread_client:
            gsheet_url = st.sidebar.text_input("Google Sheet URL'sini yapıştırın")
            if st.sidebar.button("Veriyi Çek"):
                with st.spinner("Google Sheets verisi okunuyor..."):
                    gsheet_data = load_from_gsheets(gspread_client, gsheet_url)
                    df, error = validate_and_load_data("Google Sheets", gsheet_data)

    if df is None:
        st.info("Lütfen analize başlamak için kenar çubuğundan geçerli bir veri kaynağı sağlayın.")
        return

    st.title(f"🚀 {subscription_plan} Finansal Analiz Paneli")
    analiz = calistir_analiz(df)
    if "hata" in analiz:
        st.error(f"Analiz hatası: {analiz['hata']}"); return

    if subscription_plan == 'Uzman':
        st.sidebar.header("2. Raporlama")
        
        # PDF oluşturmak için gerekli tüm verileri topla
        model, tahmin = prophet_tahmini_yap(analiz['aylik_veri'])
        forecast_fig = None
        stratejik_yorum = "Tahmin için yeterli veri yok."
        if model and tahmin is not None:
            forecast_fig = plot_plotly(model, tahmin, xlabel="Tarih", ylabel="Gelir")
            if api_key:
                stratejik_yorum = tahmin_yorumu_uret(api_key, tahmin)

        pdf_bytes = generate_pdf_report(analiz, stratejik_yorum, forecast_fig)
        if pdf_bytes: # Sadece PDF başarıyla oluşturulduysa butonu göster
            st.sidebar.download_button(
                label="PDF Raporu İndir",
                data=pdf_bytes,
                file_name=f"KazKaz_Finansal_Rapor_{time.strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )

    if analiz['kar_marji'] < 15:
        st.warning(f"⚠️ Kritik Eşik Uyarısı: Kar marjınız (%{analiz['kar_marji']:.2f}) %15'in altında. Maliyetleri gözden geçirin.", icon="🚨")

    tabs = ["Genel Bakış"]
    if subscription_plan in ['Pro', 'Uzman']:
        tabs.extend(["Gelir Analizi", "Gider Analizi"])
    if subscription_plan == 'Uzman':
        tabs.append("Gelecek Tahmini")

    tab_objects = st.tabs(tabs)

    with tab_objects[0]:
        st.header("Genel Finansal Durum")
        skor = max(0, min(100, analiz['kar_marji'] * 2.5))
        st.plotly_chart(create_gauge_chart(skor, "Finansal Sağlık Skoru"), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(analiz['fig_bar'], use_container_width=True)
        with col2:
            st.plotly_chart(analiz['fig_line'], use_container_width=True)

    if 'Gelir Analizi' in tabs:
        with tab_objects[tabs.index('Gelir Analizi')]:
            st.header("Detaylı Gelir Analizi")
            col1, col2 = st.columns(2)
            with col1:
                 if analiz['fig_urun']: st.plotly_chart(analiz['fig_urun'], use_container_width=True)
                 else: st.info("Gelir getiren ürün/hizmet verisi bulunamadı ('Satilan_Urun_Adi' sütununu kontrol edin).")
            with col2:
                st.plotly_chart(analiz['fig_marj'], use_container_width=True)
            if api_key and not analiz['top_urunler'].empty:
                prompt_data = f"En karlı ürün '{analiz['top_urunler'].index[0]}' ve kar marjı trendi."
                st.info(f"**AI Yorumu:** {yorum_uret(api_key, prompt_data)}")

    if 'Gider Analizi' in tabs:
        with tab_objects[tabs.index('Gider Analizi')]:
            st.header("Detaylı Gider Analizi")
            if analiz['fig_pie']:
                st.plotly_chart(analiz['fig_pie'], use_container_width=True)
                if api_key:
                    prompt_data = f"En büyük gider kalemi '{analiz['en_yuksek_gider_kategorisi']}'. Bu giderin toplamdaki payı."
                    st.info(f"**AI Yorumu:** {yorum_uret(api_key, prompt_data)}")
            else:
                st.info("Gider verisi bulunamadı.")


    if 'Gelecek Tahmini' in tabs:
        with tab_objects[tabs.index('Gelecek Tahmini')]:
            st.header("AI Destekli Gelecek Tahmini (Uzman Paket)")
            aylik_gelir = df.set_index('Tarih')[['Gelir']].resample('M').sum()
            model, tahmin = prophet_tahmini_yap(aylik_gelir)
            if model and tahmin is not None:
                fig_prophet = plot_plotly(model, tahmin, xlabel="Tarih", ylabel="Gelir")
                st.plotly_chart(fig_prophet, use_container_width=True)
                
                st.divider()
                st.subheader("🤖 Stratejik Tahmin Analizi")
                if api_key:
                    with st.spinner("AI stratejistiniz geleceği yorumluyor..."):
                        if 'stratejik_yorum' not in st.session_state:
                             st.session_state.stratejik_yorum = tahmin_yorumu_uret(api_key, tahmin)
                        st.markdown(st.session_state.stratejik_yorum)

                        st.write("---")
                        st.write("**Bu yorum faydalı oldu mu?**")
                        fb_col1, fb_col2, fb_col3 = st.columns([1,1,5])
                        if fb_col1.button("👍 Evet"):
                            log_feedback(db, user_info['uid'], 'positive', st.session_state.stratejik_yorum)
                        if fb_col2.button("👎 Hayır"):
                             log_feedback(db, user_info['uid'], 'negative', st.session_state.stratejik_yorum)
                else:
                    st.warning("Stratejik yorumu görmek için lütfen API anahtarınızı girin.")
            else:
                st.warning("Tahmin oluşturmak için yeterli veri yok (en az 2 aylık veri gereklidir).")

def show_subscription_page(db, user_info):
    """Kullanıcı için abonelik paketlerini gösterir."""
    st.title("Size En Uygun Paketi Seçin")
    st.markdown("Finansal verilerinizden en iyi şekilde yararlanmak için aboneliğinizi yükseltin.")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header("Temel")
        st.subheader("Ücretsiz")
        st.markdown("- ✅ Genel Finansal Bakış\n- ✅ Aylık Gelir/Gider Grafiği\n- ✅ Net Kâr Trendi")
        st.button("Mevcut Plan", disabled=True)

    with col2:
        st.header("Pro")
        st.subheader("₺750 / ay")
        st.markdown("- ✅ **Tüm Temel Özellikler**\n- ✅ Detaylı Gelir Analizi\n- ✅ Detaylı Gider Analizi\n- ✅ Basit AI Yorumları")
        if st.button("Pro Pakete Geç", type="primary"):
            db.collection('users').document(user_info['uid']).update({'subscription_plan': 'Pro'})
            st.rerun()

    with col3:
        st.header("Uzman")
        st.subheader("₺1500 / ay")
        st.markdown("- ✅ **Tüm Pro Özellikler**\n- ✅ AI Destekli Gelecek Tahmini\n- ✅ **Derinlemesine Stratejik AI Analizi**\n- ✅ **PDF Rapor İndirme**")
        if st.button("Uzman Pakete Geç", type="primary"):
            db.collection('users').document(user_info['uid']).update({'subscription_plan': 'Uzman'})
            st.rerun()

def main():
    if 'user_info' not in st.session_state: st.session_state['user_info'] = None
    
    firebase_ok = init_firebase()
    if not firebase_ok:
        st.warning("Firebase bağlantısı kurulamadı. Lütfen 'firebase-key.json' dosyasını veya Streamlit secrets ayarlarını kontrol edin.")
        st.stop()
        
    db = firestore.client()

    with st.sidebar:
        st.sidebar.title("KazKaz AI")
        if st.session_state.get('user_info'):
            st.write(f"Hoş Geldin, {st.session_state['user_info']['email']}")
            if st.button("Çıkış Yap"):
                for key in list(st.session_state.keys()):
                    if key != 'user_info':
                        del st.session_state[key]
                st.session_state.clear(); st.rerun()
        else:
            st.write("Finansal geleceğinize hoş geldiniz.")

    if st.session_state.get('user_info'):
        user_info = st.session_state['user_info']
        user_doc = db.collection('users').document(user_info['uid']).get()
        
        if user_doc.exists:
             user_info['subscription_plan'] = user_doc.to_dict().get('subscription_plan', 'Temel')
        else:
            db.collection('users').document(user_info['uid']).set({'subscription_plan': 'Temel', 'email': user_info['email']})
            user_info['subscription_plan'] = 'Temel'
        
        if user_info['subscription_plan'] in ['Temel', 'Pro', 'Uzman']:
            api_key = get_gemini_api_key() if user_info['subscription_plan'] in ['Pro', 'Uzman'] else None
            show_dashboard(user_info, api_key, db)
        else:
            show_subscription_page(db, user_info)

    else:
        choice = st.selectbox("Giriş Yap / Kayıt Ol", ["Giriş Yap", "Kayıt Ol"])
        st.title("Finansal Analiz Paneline Hoş Geldiniz")
        email = st.text_input("E-posta")
        password = st.text_input("Şifre", type="password")

        if choice == "Giriş Yap":
            if st.button("Giriş Yap", type="primary"):
                try:
                    user = auth.get_user_by_email(email)
                    st.session_state['user_info'] = {'uid': user.uid, 'email': user.email}; st.rerun()
                except Exception as e: 
                    st.error("E-posta veya şifre hatalı.")
        
        elif choice == "Kayıt Ol":
            if st.button("Kayıt Ol", type="primary"):
                try:
                    user = auth.create_user(email=email, password=password)
                    st.session_state['user_info'] = {'uid': user.uid, 'email': user.email}
                    st.success("Kaydınız başarıyla oluşturuldu! Panele yönlendiriliyorsunuz.")
                    time.sleep(2)
                    st.rerun()
                except Exception as e:
                    st.error(f"Kayıt sırasında bir hata oluştu: {e}")

if __name__ == '__main__':
    main()
