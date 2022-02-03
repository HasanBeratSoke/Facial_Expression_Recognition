# Yüz ifadesi tanıma projesi
## Araçlar
```
Keras
Flask
Opencv
```

<p float="left">
 <div> <img align="center" width="100" height="100" src="https://github.com/HasanBeratSoke/Facial_Expression_Recognition/blob/main/git-foto/keras.png"> </div>

<img align="center" width="100" height="100" src="https://github.com/HasanBeratSoke/Facial_Expression_Recognition/blob/main/git-foto/flask.png">

<img align="center" width="100" height="100" src="https://github.com/HasanBeratSoke/Facial_Expression_Recognition/blob/main/git-foto/opencv.png">
</p>







---  

 > Kullandığım veri setini [buradan](https://www.kaggle.com/msambare/fer2013) ulaşabilirsiniz.

* Veri seti 48x48 piksel büyüklüğünde gri resimler içermektedir. Ayrıca 7 ayrı kategoriye sahiptir. 
* `(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)`

* Veri seti yaklaşık 36bin resim içermektedir.

- verilerin %80 train %20 si test klasöründedir.

---

## Verilerilerin dagilimi 
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
![Train resimleri](https://github.com/HasanBeratSoke/Facial_Expression_Recognition/blob/main/git-foto/Train%20resimleri.png)

---
## veri setinin kategorilerine göre dagılımı

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

## Veri üreteçi kullanmak

`from tensorflow.keras.preprocessing.image import ImageDataGenerator`

- ImageDataGnerator, kerasın derin öğrenme için görüntüverilerinin ardaşık dezenlenmesi için başvurduğu sınıftır.
- Yerel dosya sistemimize kolay erişim ve farklı yapılardan veri yüklemek için birden fazla farklı yöntem sağlar.
- Oldukça güçlü veri ön işleme ve artırma yeteneklerine sahiptir.

`shuffle`: zorunlu degil ama her bir grupda rasgele goruntuleri secip secmeyecegini soyler

`batch_size` : her bir egitim verisi grubuna dahil edilecek goruntu sayisi

---

## Evrisimli sinir aglarini kurma asamasi
Kisaca bahsetmek gerekirse, 
goruntu islemede kullanilan, icerisinde bircok cesitli katman bulunan sinir agidir.
![cnn layer](https://github.com/HasanBeratSoke/Facial_Expression_Recognition/blob/main/git-foto/cnn%20layer.png)

![layer](https://github.com/HasanBeratSoke/Facial_Expression_Recognition/blob/main/git-foto/layers.png)


**Convolution Layer** : ozellikleri saptamak icin kullanilir.

![gif2](https://github.com/HasanBeratSoke/Facial_Expression_Recognition/blob/main/git-foto/gif2.gif)

**Non-Linearity Layer** : sisteme dogrusal olmayanligin yani non-linearity tanitilmasi.


**Flattening Layer** : modelin egitilmesi icin verileri hazirlar duzlestirir. 


**Pooling Layer** : agirlik sayisini azaltir ve uygunlugunu kontrol eder.

![gif1](https://github.com/HasanBeratSoke/Facial_Expression_Recognition/blob/main/git-foto/gif1.gif)

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
- activasyon fonksiyoun olarak Tanh veya sigmoid fonksiyonu kullanabilirdik ama ReLu cogu durumda daha iyi performans verdigi icin ReLu kullandim.

- `Batch normalization` : Agin icinde islemler sonucunda verileri dagilimini degistiyor.


- `Dropout` : bazi dugumlerin agirliklarini kisitlayarak overfitting azatmaya yardimci olur.
![Dropout](https://github.com/HasanBeratSoke/Facial_Expression_Recognition/blob/main/git-foto/Droput.jpg)

---

## modelin egitimi
```
epochs = 50

from tensorflow.keras.callbacks import ModelCheckpoint
#filepath = ('')

checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
```

```
history = model.fit_generator(generator=train_generator,
 steps_per_epoch=train_generator.n//train_generator.batch_size,
 epochs=epochs,
 validation_data = test_generator,
 validation_steps = test_generator.n//test_generator.batch_size,  callbacks=callbacks_list
 )
 ```

cikti:

```WARNING:tensorflow:From <ipython-input-24-5e7a18b22159>:1: Model.fit_generator (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
Instructions for updating:
Please use Model.fit, which supports generators.
Epoch 1/50
224/224 [==============================] - ETA: 0s - loss: 2.0477 - accuracy: 0.2298WARNING:tensorflow:Can save best model only with val_acc available, skipping.
224/224 [==============================] - 608s 3s/step - loss: 2.0477 - accuracy: 0.2298 - val_loss: 1.7448 - val_accuracy: 0.3096
Epoch 2/50
224/224 [==============================] - ETA: 0s - loss: 1.8305 - accuracy: 0.2950WARNING:tensorflow:Can save best model only with val_acc available, skipping.
```


**modelin egitimi yaklasik 9 saat surmustur**

---

### Modelin sonuclari

![his](https://github.com/HasanBeratSoke/Facial_Expression_Recognition/blob/main/git-foto/his.png)

* 20 epoch dan sonra train ve validation arasindaki fark iyice acilmistir yani overfittig bi gostergesidir, buna cozum olarak belki earlystop uygulanabilirdi. 
* grafik cizgisinin bazi yerlerinde koseli olmasinin sebebi dropuot uygulanmasindan dolayidir.

#### yararlandigim rehberler
- https://towardsdatascience.com/from-raw-images-to-real-time-predictions-with-deep-learning-ddbbda1be0e4
- https://www.tensorflow.org/tutorials/keras/save_and_load#save_checkpoints_during_training
- https://medium.com/@tuncerergin/convolutional-neural-network-convnet-yada-cnn-nedir-nasil-calisir-97a0f5d34cad
- http://derindelimavi.blogspot.com/2018/01/bir-derin-ogrenme-deneyi-dropout-ve.html
- https://github.com/log0/video_streaming_with_flask_example
- https://www.tensorflow.org/api_docs/python/tf/keras/models/model_from_json


*iyi calismalar :)*