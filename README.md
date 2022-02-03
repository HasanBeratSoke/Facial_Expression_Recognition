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

### Verilerilerin dagilimi 
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

`shuffle`: zorunlu degil ama her bir grupda rasgele goruntuleri secip secmeyecegini soyler

`batch_size` : her bir egitim verisi grubuna dahil edilecek goruntu sayisi

---

#### Evrisimli sinir aglarini kurma asamasi
Kisaca bahsetmek gerekirse, 
goruntu islemede kullanilan, icerisinde bircok cesitli katman bulunan sinir agidir.

**Convolution Layer** : ozellikleri saptamak icin kullanilir.

**Non-Linearity Layer** : sisteme dogrusal olmayanligin yani non-linearity tanitilmasi.

**Flattening Layer** : modelin egitilmesi icin verileri hazirlar duzlestirir. 

**Pooling Layer** : agirlik sayisini azaltir ve uygunlugunu kontrol eder.

prejemizdeki kullandigimiz agin mimarisi:
```
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam


# Initialising the CNN
model = Sequential()

# 1 - Convolution
model.add(Conv2D(64,(3,3), padding='same', input_shape=(48, 48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2nd Convolution layer
model.add(Conv2D(128,(5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 3rd Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 4th Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flattening
model.add(Flatten())

# Fully connected layer 1st layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Fully connected layer 2nd layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(nb_classes, activation='softmax'))

opt = Adam(lr=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
```
