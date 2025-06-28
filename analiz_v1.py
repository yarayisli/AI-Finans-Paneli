# Gerekli olan tek kütüphanemizi, yani pandas'ı projemize dahil ediyoruz.
# Veri analizini bu kütüphane sayesinde yapacağız.
import pandas as pd

def ana_analiz_motoru(dosya_yolu):
    """
    Bu fonksiyon, projemizin kalbidir.
    Veri dosyasını okur, analiz eder ve sonuçları anlaşılır bir şekilde sunar.
    """
    try:
        # Adım 1: CSV dosyasını oku ve bir DataFrame'e (veri çerçevesi) dönüştür.
        # DataFrame, verilerimizi satır ve sütunlardan oluşan bir tablo gibi düşünmemizi sağlar.
        veri = pd.read_csv(dosya_yolu)
        print("✓ Veri dosyası başarıyla okundu.")

        # Adım 2: Temel Finansal Hesaplamalar
        # Toplam gelir ve gideri, ilgili sütunların toplamını alarak hesaplıyoruz.
        toplam_gelir = veri['Gelir'].sum()
        toplam_gider = veri['Gider'].sum()
        # Kar, gelir ile gider arasındaki farktır.
        net_kar = toplam_gelir - toplam_gider

        # Adım 3: En Yüksek Gider Kalemini Bulma
        # Giderleri kategorilerine göre gruplayıp her kategorinin toplamını alıyoruz.
        gider_kategorileri = veri.groupby('Kategori')['Gider'].sum()
        # En yüksek değerli kategoriyi buluyoruz.
        en_yuksek_gider_kategorisi = gider_kategorileri.idxmax()
        en_yuksek_gider_tutari = gider_kategorileri.max()

        # Adım 4: Ürün Bazında Karlılık Analizi
        # Her bir ürün için toplam geliri hesaplıyoruz.
        urun_gelirleri = veri.groupby('Satilan_Urun_Adi')['Gelir'].sum()
        # En çok gelir getiren ürünü buluyoruz.
        en_cok_gelir_getiren_urun = urun_gelirleri.idxmax()
        
        print("--- FİNANSAL ÖZET ---")
        print(f"Toplam Gelir: {toplam_gelir} TL")
        print(f"Toplam Gider: {toplam_gider} TL")
        print(f"Net Kar: {net_kar} TL")
        print("-" * 20) # Ayraç çizgisi
        print("--- ANALİZ SONUÇLARI ---")
        print(f"En Yüksek Gider Kalemi: '{en_yuksek_gider_kategorisi}' kategorisi ({en_yuksek_gider_tutari} TL)")
        print(f"En Çok Gelir Getiren Ürün: '{en_cok_gelir_getiren_urun}'")
        print("-" * 20)
        
        # Adım 5: "İlkel" Öneri Motoru
        # Hesapladığımız sonuçlara göre basit kurallar çalıştırıyoruz.
        print("--- ÖNERİLER ---")
        if net_kar < 0:
            print("⚠️ DİKKAT: Şirket şu anda zarar ediyor. Gider kalemleri acilen incelenmeli.")
        elif net_kar / toplam_gelir < 0.10: # Kar marjı %10'dan düşükse
            print("💡 İYİLEŞTİRME FIRSATI: Kar marjı düşük görünüyor. Maliyetleri düşürme veya fiyatlandırmayı gözden geçirme düşünülebilir.")
        else:
            print("👍 Gidişat pozitif. Karlılığı korumak için maliyet kontrolüne devam edin.")

        if en_yuksek_gider_tutari / toplam_gider > 0.5: # Tek bir kategori toplam giderin yarısından fazlaysa
            print(f"💡 İYİLEŞTİRME FIRSATI: Giderlerin büyük bir kısmı '{en_yuksek_gider_kategorisi}' kategorisine gidiyor. Bu alanda tasarruf potansiyeli olabilir.")
            
    except FileNotFoundError:
        # Eğer belirtilen yolda dosya bulunamazsa verilecek hata mesajı.
        print(f"HATA: '{dosya_yolu}' adında bir dosya bulunamadı. Dosyanın doğru klasörde olduğundan emin misin?")
    except Exception as e:
        # Beklenmedik başka bir hata olursa bunu göster.
        print(f"Beklenmedik bir hata oluştu: {e}")

# --- Script'in Çalıştırıldığı Ana Bölüm ---
if __name__ == "__main__":
    print("Yapay Zeka Finansal Analiz Motoru v0.1 Başlatılıyor...")
    # Analiz edilecek dosyanın adını buraya yazıyoruz.
    # Bu script ile aynı klasörde olmalı.
    dosya_adi = "ornek_veri.csv"
    ana_analiz_motoru(dosya_adi)
    print("\nAnaliz tamamlandı.")