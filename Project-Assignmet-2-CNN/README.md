## MNIST Veri Seti Üzerinde Farklı CNN Mimarilerini Deneme Proje Ödevi

MNIST veri seti üzerinde farklı derin öğrenme stratejilerinin performansını ve teorik sınırlarını analiz edilmiştir

## 1. Giriş (Introduction)
Bu projenin temel odağı, model derinliğinin, optimizasyon tekniklerinin ve özellik çıkarımı (feature extraction) stratejilerinin el yazısı rakam tanıma (MNIST) başarısı üzerindeki etkisini ölçmektir. Çalışmada klasik evrişimli sinir ağları (LeNet-5) ile derin mimariler (VGG11) karşılaştırılmış, aynı zamanda makine öğrenmesi ve derin öğrenmeyi birleştiren hibrit bir yaklaşımın sınırları test edilmiştir.

## 2. Yöntem (Method)
Çalışma, tekrarlanabilirlik ilkesine uygun olarak **PyTorch** framework'ü üzerinde, **NVIDIA Tesla T4 GPU** kullanılarak koşturulmuştur. Modellerin tamamı hızlı bir benchmarking yapabilmek adına 2 epoch boyunca eğitilmiştir.

### 2.1. Veri Seti ve Ön İşleme
* **Dataset:** MNIST (60,000 Eğitim / 10,000 Test görseli).
* **Pre-processing:** Görseller 28x28 boyutunda tensörlere dönüştürülmüş ve PyTorch standartlarına uygun olarak normalize edilmiştir.

### 2.2. Model Mimarileri ve Stratejiler
* **LeNet-5:** Klasik mimari ve üzerine eklenen BatchNorm/Dropout katmanlarıyla modifiye edilmiş versiyonları.
* **VGG11:** Derin mimarinin sıfırdan (Scratch) ve Transfer Learning (ImageNet weights) yöntemleriyle kıyası.
* **Hybrid Approach:** VGG11'in son katmanının `nn.Identity` ile değiştirilmesiyle elde edilen 4096 boyutlu özelliklerin `.npy` formatında diske kaydedilmesi ve **Random Forest** algoritması ile sınıflandırılması.

## 3. Sonuçlar (Results)
Modellerin 2 epoch sonunda elde ettiği test seti doğruluk oranları aşağıda listelenmiştir:

| Rank | Model Architecture & State | Test Accuracy (%) |
| :--- | :--- | :---: |
| 1 | **VGG11 (Trained from scratch)** | **98.51** |
| 2 | LeNet-5 (Trained) | 97.50 |
| 3 | LeNet-5 Modified (BatchNorm + Dropout) | 96.74 |
| 4 | VGG11 (Transfer Learning) | 93.67 |
| 5 | **Hybrid (Untrained VGG11 + Random Forest)** | **91.01** |
| 6 | LeNet-5 (Untrained) | 10.10 |

> **Görselleştirme:** > *(Not: Aşağıdaki dosya yollarını bilgisayarındaki grafik/matris görsellerinin adlarıyla değiştir)* > ![Training Loss](assets/loss_plot.png)  
> *Şekil 1: Modellerin epoch bazlı eğitim kaybı (loss) değişimi.* >  
> ![Confusion Matrix](assets/confusion_matrix.png)  
> *Şekil 2: En yüksek doğruluğa sahip VGG11 (Scratch) modeli için karmaşıklık matrisi.*

## 4. Tartışma (Discussion)
Elde edilen verilere dayalı teknik analizler şöyledir:

* **Scratch vs. Transfer Learning:** Sıfırdan eğitilen VGG11 (%98.51), ImageNet ağırlıklarıyla önceden eğitilmiş Transfer Learning modelini (%93.67) geride bırakmıştır. Bunun temel teorik sebebi, ImageNet'in 1000 sınıflı, renkli ve karmaşık (kedi, araba vb.) görsellerden oluşmasıdır. Modelin dondurulmuş katmanlarındaki filtreler bu karmaşık dokulara ayarlıyken, MNIST gibi basit, siyah-beyaz çizgi ve kenarlardan oluşan bir veriye sadece sınıflandırıcı (classifier) katmanını eğiterek 2 epoch'ta adapte olmak yetersiz kalmıştır.
* **Hibrit Modelin Verimliliği:** Tamamen eğitimsiz (rastgele ağırlıklara sahip) bir VGG11 ağının çıkardığı özelliklerin Random Forest ile sınıflandırılması %91.01 gibi etkileyici bir sonuç vermiştir. Buna karşılık eğitimsiz LeNet-5 sadece %10.10 (rastgele tahmin) yapabilmiştir. Bu durum, derin evrişim katmanlarının rastgele filtrelerle bile veriyi 4096 boyutlu devasa bir uzaya ayırıcı bir şekilde izdüşürebildiğini ve Random Forest gibi güçlü bir algoritmanın bu kaotik veriden bile anlamlı bir örüntü yakalayabildiğini teorik olarak kanıtlamaktadır.
* **Düzenlileştirme Etkisi:** LeNet-5 Modifiye modelinin (%96.74), klasik LeNet-5'in (%97.50) ufak bir farkla gerisinde kalması beklenen bir durumdur. Ağa eklenen Dropout, nöronları rastgele kapattığı için modelin ilk başlarda öğrenme hızını yavaşlatır. Eğitim sadece 2 epoch ile sınırlandığı için Dropout'lu model tam potansiyeline ulaşamamıştır; daha uzun epoch'larda (örneğin 10-20 epoch) modifiye edilmiş ağın aşırı öğrenmeyi (overfitting) engelleyerek klasik modeli geçmesi beklenir.
* **Maliyet/Performans:** LeNet-5 modeli %97.50 doğruluğa ulaşırken, kendisinden katbekat daha fazla parametreye ve işlem yüküne sahip olan VGG11 %98.51 doğruluğa ulaşabilmiştir. Sadece ~%1'lik bir performans artışı için donanım maliyetinin bu kadar artırılması, MNIST gibi görece "kolay" veri setlerinde LeNet-5 gibi hafif mimarilerin aslında çok daha verimli ve yeterli olduğunu göstermektedir.
