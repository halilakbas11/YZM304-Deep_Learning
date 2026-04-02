# YZM304 Derin Öğrenme Dersi 
## I. Proje Ödevi: [MLP Kullanılarak Penguenlerin Fiziksel Özelliklerinden Türlerinin Sınıflandırılması]

## 1. Introduction (Giriş)
<div align="justify">
  
* **Problem Tanımı:** Bu proje ödevi için problemimiz MLP kullanarak ikili/çoklu sınıflandırma yapabilmektir. MLP mimarisi için hem laboratuvar saatlerinde gösterilen baştan yazım hem de hazır kütüphanelerden yararlanılması istenmiştir.

* **Motivasyon:** Bu çalışma ile MLP yapısının temel fonksiyonları ve temel işleyiş biçimi, python üzerinden kodlanarak matemaiksel bakış açışıyla incelenmiştir. Aynı şekilde MLP için kullanılan hiperparameterlerin, optimizerların değişmesinin sonuç üzerindeki etkisi incelenmiştir.

* **Kısa Özet:** Bu projede kısaca, "Penguin Sizes Dataset" veri seti kullanılarak veri setindeki 3 farklı penguen türünü tahmin etmeye yönelik çoklu sınıf çıktılı bir MLP geliştirilmiştir. Proje ilk olarak veri ön işleme kısmı ile başlamaktadır. Bu kısım; eksik verilerin IterativeImputer ile doldurulması, kategorik değişkenlerin encoding ile dönüştürülmesi, sayısal özelliklerin normalleştirilmesini, veri setinin train/val/test şeklinde bölünmesini içermektektedir. Ardından, NumPy kütüphanesi kullanılarak sıfırdan (2 katmanlı - 1 gizli katmanlı) MLP modeli kodlanmış ve çoklu sınıflandırma için çıktı katmanına Softmax fonksiyonu modifiyesi yapılmıştır. Model, çeşitli hiperparametre kombinasyonları (hidden layer nöron sayıları, öğrenme oranları) ve optimizasyon algoritmaları (SGD, BGD) ile eğitilerek performans değişimleri gözlemlenmiştir. Elde edilen bulgular, Scikit-learn kütüphanesine ait MLPClassifier modeli ile karşılaştırmalı olarak analiz edilerek son bir karşılaştırma tablosu oluşturulmuştur.

</div>

## 2. Methods (Methods / Yöntemler)
<div align="justify">
  
* **Veri Seti ve Ön İşleme:**
  * Kullanılan veri seti: Kaggle'da bulunan "Penguin Sizes Dataset" veri seti kullanılmıştır. Veri setinde penguenlere ait tür, yaşam alanı ve vücut fiziksel özellikleri bilgileri verilmektedir. Veri seti (344,7) boyutlarındadır. Target sütunu "Species" yani tür seçilmiştir fakat bu durum sınıf dengesizliği olan bir veri seti oluşturmuştur (Adelie:152, Gentoo:124, Chinstrap:68).

   * Veri temizleme adımları: İlk olarak veri seti üzerinde kaba bir inceleme yapılmıştır. Sonrasında her sütun için null sayıları tespit edilmiştir, bunların arasından sayısal olmayan cinsiyet sütunu için ilk olarak mapping (male:0, female:1) yapılmıştır. Ardından, sklearn üzerinden "NaN" olan değerlerin sütunu target seçilerek diğer sütunları eğitim kullanan ML modeli "Imputer" kullanarak doldurulmuştur. Burada cinsiyet sütununu bu şekilde doldurmak biraz unorthodox bir yöntem olsa da 0/1 olduğu için NaN değerleri doldurulduktan sonra sayısal yuvarlama yapılarak 0 ve 1 değerlerine getirilmiştir. Aktivasyon fonksiyonu olarak ReLU kullanılacağı için eğitim setine (X) "StandartScaler" ön işlemesi yapılmıştır. Modelin sayısal büyüklüğü karıştırmaması için ve Softmax ile (Categorical Cross-Entropy loss) kullanılabilmesi için "Species" hedef sütununa "one-hot encoding" yapılarak "olasılık/vektör" formatına getirilmiştir ve veri ön işleme aşaması sonlandırılmıştır.
* **Model Mimarisi:**
  * Katman sayıları ve her katmandaki nöron sayısı: 2 katmanlı - 1 gizli katmanlı model. Gizli katman için varsayılan n_h = 6 kullanılmakla beraber deneyler için "9, 15" de denenmiştir.

  * Çoklu sınıflandırma model yapısı: Girdi Özellikleri -> [Gizli Katman: ReLU] -> [Çıktı Katmanı: Softmax] -> Sınıf Olasılıkları (3 Penguen Türü)

* **Başlangıç Ayarları ve Hiperparametreler (Tekrarlanabilirlik İçin):**
  * Ağırlık başlatma (Weight Initialization) yöntemi: initialize_parameters fonksiyonu içerisinde He Initialization.
  * train/val/test bölme oranı: train/test: 0.6/0.4 - > train/val/test : 0.48/0.12/0.4
  * Random Seed: 42
  * Hiperparametreler: Batch GD için varsayılanlar, "n_h=6, n_steps=500, learning_rate=0.01". Stochastic GD için varsıylanlar, "n_h=9, n_steps=50, learning_rate=0.0025" (NOT: Kodda bu şekilde gözükmüyor). Sklearn-Adam için varsayılanlar, n_h=6, n_steps=1000, l_r=0.05, alpha=0.001 (NOT: kodda run_MLP_sklearn fonknsiyonu içerisinde verilen değerler bunlarla farklıdır, bu değerler denemeler sonucunda optimal olacak şekilde oluşturulmuştur, SGD için de aynı durum yaşanmıştır.)
  * Optimizasyon algoritmaları: Batch GD (Batch Gradient Descent), SGD (Stochastic Gradient Descent), Adam (Adaptive Moment Estimation).
  * Kayıp fonksiyonu (Loss function): compute_cost fonksiyonu içerisinde Categorical Cross-Entropy kayıp fonksiyonu kullanılmıştır.
  
* **Kod Yapısı:**
<div align="center">
  
<img width="431" height="325" alt="Screenshot 2026-04-02 142436" src="https://github.com/user-attachments/assets/77dd2b39-f73d-4c6d-9d6e-4922ffd89aa9" />
</div>

## 3. Results (Sonuçlar)
<div align="center">

### Model Optimizasyon ve Karşılaştırma Sonuçları

| Deney Adı | Gizli Katman Nöronu (n_h) | Epoch (n_steps) | Öğrenme Oranı ($\alpha$) | Optimizasyon Algoritması | F1-Macro Skoru |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **exp2_1** | 6 | 501 | 0.0100 | SGD | 1.0000 |
| **exp3_5** | 9 | 45 | 0.0500 | Sklearn - Adam+L2+EarlyStop | 1.0000 |
| **exp1_4** | 6 | 501 | 0.0500 | BGD | 0.9939 |
| **exp2_2** | 6 | 15 | 0.0100 | SGD | 0.9939 |
| **exp2_4** | 9 | 50 | 0.0025 | SGD | 0.9939 |
| **exp2_3** | 6 | 50 | 0.0025 | SGD | 0.9939 |
| **exp1_3** | 6 | 1500 | 0.0100 | BGD | 0.9851 |
| **exp1_1** | 6 | 501 | 0.0100 | BGD | 0.9584 |
| **exp3_4** | 6 | 47 | 0.0500 | Sklearn - Adam+L2+EarlyStop | 0.8749 |
| **exp1_2** | 9 | 501 | 0.0100 | BGD | 0.8155 |
| **exp3_3** | 9 | 80 | 0.0010 | Sklearn - Adam+L2+EarlyStop | 0.5503 |
| **exp3_1** | 6 | 33 | 0.0010 | Sklearn - Adam+L2+EarlyStop | 0.1868 |
| **exp3_2** | 6 | 33 | 0.0010 | Sklearn - Adam+L2+EarlyStop | 0.1868 |

<br>
</div>

<div align="center">
<img width="1059" height="374" alt="image (1)" src="https://github.com/user-attachments/assets/bfcd8766-19c4-4d0f-970c-5cd95dfb14f9" />
</div>
Baştan yazdığım model yaptığım ilk testti. Grafikten ve karmaşıklık matrisinden (cm) görülebileceği üzere model, veri dengesizliğine rağmen iyi bir tahminde bulunabilmiştir. Loss grafiğinde 200. iterasyondan sonra overfitting izleri görülse bile genelleme başarısının yeterli olduğunu cm üzerinden görebilmekteyiz.
<br>
<div align="center">
<img width="1059" height="374" alt="image" src="https://github.com/user-attachments/assets/59539a74-b0a6-4f3f-9418-179d8f87b1b1" />
</div>
SGD'li modelin 2. testi, 1. testte görülen hızlı yakınsamayı çözmek için iterasyon sayısı 501-> 15 yapılmıştı. İki loss grafiği arasındaki boşluğa bakarak modelin overfit olmadığı, genelleme yeteneğini koruduğunu söylebiliriz. Ve buna rağmen F1 skor 0.99 gelmiştir. Bu da modelin çok başarılı olduğunu göstermektedir. Veri yetersizliği nedeniyle modelin genelleme yeteneğini tam ölçememiş olsam bile grafikten veri sayısı artırılsaydı yine başarılı olacağını görebilyoruz.
<br>
<div align="center">
<img width="1083" height="374" alt="image (2)" src="https://github.com/user-attachments/assets/52429248-1c79-4d11-890b-17cc15f250e1" />
</div>
Sklearn modeli ve Adam optimizer kullanılarak yapılan tek olumlu sonuç. Model "Adelie" ve "Chinstrap" türleri arasındaki ayrımı yakalayamamış ve bu da yüksek validasyon hatasına yol açmıştır. Dolayısıyla f1 skoru da diğerlerine kıyasla düşük gelmiştir. 

Bu deneyden sonra yapılan 5.deneyde exp3_5te model F1: 1.0 olağanüstü performansını göstermiştir fakat grafikte sonlara doğru overfit kuşkusu vardır. Ekstra veri ile test yapmak gereklidir.
<br>
<div align="center">
<img width="789" height="490" alt="download" src="https://github.com/user-attachments/assets/2101761d-10cc-4204-abf4-0934845e9225" />
</div>

## 4. Discussion (Tartışma)

<div align="justify">

* **Sonuçların Yorumlanması:**
  Genel olarak, sıfırdan geliştirilen MLP modeli Penguin Size Dataset veri seti üzerinde oldukça başarılı olmuştur ama kesin güvenilir olamayan sonuçlar elde etmiştir (F1-Macro: %95 - %99 bantlarında). Kendi yazdığımız temel BGD ve SGD modelleri, veriye çok hızlı adapte olmuş ve yüksek doğruluk oranlarına ulaşmıştır. Özellikle SGD algoritmasının, küçük veri setlerindeki stokastik yapısı sayesinde yerel minimumlara (local minima) takılmadan hızlıca yakınsadığı görülmüştür. Öte yandan, Scikit-learn kütüphanesiyle oluşturulan model (`exp3_4` deneyi) varsayılan hiperparametreler ve Adam optimizasyon algoritması ile başlangıçta "Adelie" ve "Chinstrap" türlerini ayırmakta zorlanmıştır. Bunun temel sebebi, "Chinstrap" türünün veri setindeki azlığı (sınıf dengesizliği) ve Adam algoritmasının küçük veri setlerinde, SGD'ye kıyasla daha karmaşık bir hiperparametre ayarlaması gerektirmesidir. Ama yine de en sonda hiperparametreler optimize edildiğinde (`exp3_5`), Sklearn modeli de F1: 1.0 skoruna ulaşmış olduğunu görebilmekteyiz.

* **Overfitting / Underfitting Analizi:**
  Çalışma boyunca veri setinin küçük olması (344 örnek) en büyük zorluklardan biri olmuş ve modelleri aşırı öğrenmeye yatkın hale getirmiştir. Örneğin ilk BGD denemesinde, 200. epoch'tan sonra eğitim hatası düşmeye devam ederken validasyon hatasının sabit kalması/artması overfitting'in başladığını net bir şekilde göstermiştir. Bu problemi çözmek için SGD modelinde epoch sayısı radikal bir şekilde düşürülerek (501'den 15'e) bir nevi manuel "Early Stopping" uygulanmış ve modelin ezber yapmadan genelleme yeteneğini koruması sağlanmıştır (F1: 0.9939). Sklearn modelinde bu durum, yerleşik Early Stopping mekanizması ve L2 Regülarizasyonu (ağırlık cezalandırması) kullanılarak algoritmik olarak kontrol altına alınmaya çalışılmıştır.

* **Gelecek Çalışmalar:**
  Modelin %100'e yakın doğruluk oranları vermesi (özellikle F1: 1.0 skoru), veri setinin kolay ayrılabilir özelliklere sahip olmasından kaynaklandığı kadar, veri azlığından dolayı oluşan "şanslı bir train/test bölünmesi" ihtimalini, penguen fiziksel özellikleri uzayının çok az bir kısmının görülmesi nedeniyle gerçek veriler başarısızlık riskini doğurmaktadır. Gelecek çalışmalarda modelin gerçek dünyadaki genelleme başarısını daha kesin ölçebilmek için ilk olarak ek veri bulunmalı ve ek olarak K-Fold Cross Validation (Çapraz Doğrulama) tekniği kullanılmalıdır. Ayrıca, azınlık sınıfı olan "Chinstrap" verilerini artırmak için SMOTE gibi sentetik veri artırma teknikleri denenebilir. Model mimarisi açısından ise, ağı derinleştirerek aralara Batch Normalization veya Dropout katmanları eklemek, olası overfitting risklerini daha başından sönümlemek için iyi birer geliştirme adımı olacaktır. Ve ekstra hidden layer katmanı eklenerek de denenmelidir.

</div>
