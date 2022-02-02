# Yüz ifadesi tanıma projesi
### Araçlar
```
Keras
Flask
Opencv
```

---  

 > Kullandığım veri setini [buradan](https://www.kaggle.com/msambare/fer2013) ulaşabilirsiniz.

* Veri seti 48x48 piksel büyüklüğünde gri resimler içermektedir. Ayrıca 7 ayrı kategoriye sahiptir. `(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)`

* Veri seti yaklaşık 36bin resim içermektedir.

- verilerin %80 train %20 si test klasöründedir.

---

### Verilerimize bi göz atalım :)
```python
pic_size = 48

base_path = "../Facial_Expression_Recognition/images/"

plt.figure(0, figsize=(12,20))
cpt = 0

for expression in os.listdir(base_path + "train/"):
    for i in range(1,6):
        cpt = cpt + 1
        plt.subplot(7,5,cpt)
        img = load_img(base_path + "train/" + expression + "/" +os.listdir(base_path + "train/" + expression)[i], target_size=(pic_size, pic_size))
        plt.imshow(img, cmap="gray")

plt.tight_layout()
plt.show()
```
![Train Resimleri]()

---
#### veri setinin kategorilerine göre dagılımı

```Python
for expression in os.listdir(base_path + 'train'):
    print(str(len(os.listdir(base_path+ 'train/' + expression)))+' '+ expression+' images')
```
Cıktı:
```
3995 angry images
436 disgust images
4097 fear images
7215 happy images
4965 neutral images
4830 sad images
3171 surprise images
```
* veri setinde kategoriler 'disgust' dışında gayet dengeli dağılmıştır.

---

#### Veri üreteçi kullanmak

`from tensorflow.keras.preprocessing.image import ImageDataGenerator`

- ImageDataGnerator, kerasın derin öğrenme için görüntüverilerinin ardaşık dezenlenmesi için başvurduğu sınıftır.
- Yerel dosya sistemimize kolay erişim ve farklı yapılardan veri yüklemek için birden fazla farklı yöntem sağlar.
- Oldukça güçlü veri ön işleme ve artırma yeteneklerine sahiptir.