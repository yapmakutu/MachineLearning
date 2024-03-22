# Gerekli kütüphaneleri içe aktar
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os

# Dosya dizinini belirt
dizin = r"C:\Users\Lenovo\Desktop\BreastCancerDataset\benign"

# Dizindeki tüm dosyaları al
dosya_listesi = os.listdir(dizin)

# Görüntü dosyalarını filtreleme (örneğin, sadece JPEG uzantılı dosyalar)
gorsel_dosyalari = [dosya for dosya in dosya_listesi if dosya.lower().endswith(('.png', '.jpg', '.jpeg'))]

# İlk görüntüyü seçme
Image = cv2.imread(os.path.join(dizin, gorsel_dosyalari[0]))

# Olası gürültüleri filtreleme
image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
salt_img_filtered = cv2.medianBlur(image, ksize=3)
gaussian_img_filtered = cv2.GaussianBlur(salt_img_filtered, (3, 3), 0)
img_filtered = cv2.bilateralFilter(gaussian_img_filtered, d=9, sigmaColor=75, sigmaSpace=75)

# Siyah-beyaz dönüşümü
_, thresholded_img = cv2.threshold(img_filtered, 127, 255, cv2.THRESH_BINARY)

# Yeni bir figür oluşturun ve filtrelenmiş görüntüyü gösterin
plt.figure(4)
plt.imshow(thresholded_img, cmap='gray')
plt.title('Threshold Uygulanmış Görüntü')
plt.show()

# Filtrelenmis image arrayin icine aktaririliyor
img = np.array(thresholded_img, dtype=np.float64)

# İlk seviye set etme (Level Set Function - LSF)
# İlk olarak, np.ones fonksiyonu kullanılarak, görüntü boyutlarına uygun bir şekilde bir dizi oluşturulur
# Bu dizi, başlangıçta tüm elemanları 1 olan bir dizi oluşturur
IniLSF = np.ones((img.shape[0], img.shape[1]), img.dtype)
#Ardından, bu dizinin belirli bir bölgesini seçer (30'dan 80'e kadar olan satırlar ve sütunlar) ve bu seçilen bölgeye -1 değerini atar
IniLSF[30:80, 30:80] = -1
# Tüm dizinin elemanlarını negatif değerlere çevir
# LSF nin seçilen bölgesi negatif değerlere sahip olur
IniLSF = -IniLSF

# Renk kanallarını düzenleme
# OpenCV'nin varsayılan renk uzayından diğer bir sık kullanılan renk uzayına dönüşüm yapar
# Görüntünün renklerini doğru şekilde görselleştirmek için python icin donusum
Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)

# Maske oluşturma
# Özelleştirilebilir koordinatlar
start_x, start_y, end_x, end_y = 250, 120, 350, 180
mask = np.zeros_like(IniLSF, dtype=np.uint8)
cv2.rectangle(mask, (start_x, start_y), (end_x, end_y), (255), thickness=cv2.FILLED)

# Maskelenmiş görüntü oluşturma
result_image = cv2.bitwise_and(Image, Image, mask=mask)

# Yardımcı matematik fonksiyonu
def mat_math(input, str):
    output = input
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # arctan
            # Vektörlerin yönünü belirlemek için kullaniyorum
            # Kenarları tespit etmek için kullaniyorum
            if str == "atan":
                output[i, j] = math.atan(input[i, j])
            # Görüntü üzerindeki mesafeleri ölçmek amacıyla kullaniyorum bir sonraki noktaya gidis icin
            # Görüntünün kontrastını artırabilir detayları daha iyi görmek icin
            if str == "sqrt":
                output[i, j] = math.sqrt(input[i, j])
    return output

# Çözüm fonksiyonu (Chan-Vese)
def CV(LSF, img, mu, nu, epison, step):
    # Aktivasyon fonksiyonu (Heaviside fonksiyonu)
    Drc = (epison / math.pi) / (epison * epison + LSF * LSF)
    Hea = 0.5 * (1 + (2 / math.pi) * mat_math(LSF / epison, "atan"))

    # Gradient hesaplamaları
    Iy, Ix = np.gradient(LSF)
    s = mat_math(Ix * Ix + Iy * Iy, "sqrt")
    Nx = Ix / (s + 0.000001)
    Ny = Iy / (s + 0.000001)
    Mxx, Nxx = np.gradient(Nx)
    Nyy, Myy = np.gradient(Ny)
    cur = Nxx + Nyy
    Length = nu * Drc * cur

    # Laplacian ve Penalty hesaplamaları
    Lap = cv2.Laplacian(LSF, -1)
    Penalty = mu * (Lap - cur)

    # Sınıf merkezi (C1 ve C2) ve CV terimi hesaplamaları
    s1 = Hea * img
    s2 = (1 - Hea) * img
    s3 = 1 - Hea
    C1 = s1.sum() / Hea.sum()
    C2 = s2.sum() / s3.sum()
    CVterm = Drc * (-1 * (img - C1) * (img - C1) + 1 * (img - C2) * (img - C2))

    # LSF'nin güncellenmesi
    LSF = LSF + step * (Length + Penalty + CVterm)
    return LSF

# Chan-Vese parametreleri
mu = 1  # Chan-Vese enerji terimi katsayısı
nu = 0.003 * 255 * 255  # Active contour modelinin uzunluk terimi katsayısı
num = 20  # İterasyon sayısı
epison = 1  # Aktivasyon fonksiyonu genişlik kontrol parametresi
step = 0.1  # Her iterasyonda LSF güncellemesi için adım boyutu
LSF = IniLSF  # Başlangıç Seviye Set Fonksiyonu (Initial Level Set Function)

# Çözümün İteratif Uygulanması
for i in range(1, num):
    LSF = CV(LSF, img, mu, nu, epison, step)
    # Active contour ile tespit edilen bölgeleri içeren bir maske oluştur
    # Zero array icine LSF value olan yerleri doldurma
    contour_mask = np.zeros_like(LSF, dtype=np.uint8)
    contour_mask[LSF > 0] = 255
    # Orijinal görüntü üzerine active contour sınırlarını ekle
    result_contour = Image.copy()
    # Active contour sınırlarını kırmızı renkte ekle
    result_contour[contour_mask > 0] = [255, 0, 0]

    # Active contour ile tespit edilen bölgelerin siyah-beyaz görüntüsünü oluştur
    result_binary = np.zeros_like(contour_mask)
    result_binary[contour_mask > 0] = 255

# İki farklı çıktıyı göster
plt.figure(1)
plt.imshow(result_contour)
plt.title('Active Contour Sınırları')
plt.show()

plt.figure(2)
plt.imshow(result_binary, cmap='gray')
plt.title('Active Contour İle Tespit Edilen Bölgeler')
plt.show()

# Belirli bir bölge seçimi için koordinatları ayarlayın
selected_area = result_binary[start_y:end_y, start_x:end_x]

# Yeni bir figür oluşturun ve seçili bölgeyi gösterin
plt.figure(3)
plt.imshow(selected_area, cmap='gray')
plt.title('Kitle Seçilen Alan')
plt.show()

# selected_area üzerinde tersleme işlemi
inverted_area = cv2.bitwise_not(selected_area)

# Yeni bir figür oluşturun ve terslenmiş bölgeyi gösterin
plt.figure(4)
plt.imshow(inverted_area, cmap='gray')
plt.title('Terslenmiş Kitleli Alan')
plt.show()
def detect_and_draw_circle(image):
    # Kenarları bulma
    edges = cv2.Canny(image, 30, 100)

    # Contour'ları bulma
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # En büyük konturun bulunması
    largest_contour = max(contours, key=cv2.contourArea)

    # Contour'un çevresine bir çember çizme
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center = (int(x), int(y))
    radius = int(radius)

    # Görüntü üzerine çemberi çizme
    result_image = image.copy()
    cv2.circle(result_image, center, radius, (0, 255, 0), 2)

    return result_image

# Seçilen bölge üzerinde yuvarlak tespiti ve çemberin çizilmiş hali
result_with_circle = detect_and_draw_circle(result_binary)

# Yeni bir figür oluşturun ve sonucu gösterin
plt.figure(5)
plt.imshow(result_with_circle, cmap='gray')
plt.title('Kitle Tespiti')
plt.show()
