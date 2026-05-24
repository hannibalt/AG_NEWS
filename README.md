# AG News Text Classification

Bu proje, popüler **AG News** veri kümesini kullanarak haber metinlerini kategorilerine (World, Sports, Business, Sci/Tech) ayırmayı amaçlayan bir metin sınıflandırma (Text Classification) projesidir. Projede hem geleneksel makine öğrenmesi hem de modern transformer mimarileri kullanılmıştır.

## 🚀 Kullanılan Modeller ve Yaklaşımlar
1. **TF-IDF + Random Forest:** Metinler kelime frekanslarına göre vektörleştirilmiş ve Random Forest algoritması ile sınıflandırılmıştır.
2. **DistilBERT (Transformers):** Hugging Face kütüphanesi kullanılarak önceden eğitilmiş DistilBERT modeli veri kümesi üzerinde ince ayar (fine-tuning) yapılmıştır.

## 🛠️ Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın
```bash
git clone [https://github.com/KULLANICI_ADIN/ag-news-classification.git](https://github.com/KULLANICI_ADIN/ag-news-classification.git)
cd ag-news-classification
