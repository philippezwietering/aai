import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10 # Want tien cijfers
epochs = 20 # Enigszins arbitraire waarde, maar lijkt goed te zijn in alle configuraties die ik gerund heb, dus maar zo gehouden

# Hier wordt de data ingeladen, op dezelfde manier als in de voorbeelden in de documentatie
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Dit verandert de numerieke waarde van de digit die het zou moeten voorstellen naar een 
# categorie zodat de pixels in een categorie kunnen worden geplaatst
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# Dit is het werkelijke neurale model
# Sequential wil zeggen dat het gaat om een simpel lineair model
# Dense geeft aan dat het gaat om een standaardlaag, dus iedere node heeft een vertakking naar alle nodes van de volgende laag
# Dropout is een laag die willekeurige waarden naar 0 zet en voorkomt overfitting van het netwerk. Het weglaten van droput lagen tussen
# de Dense lagen zorgt voor een verlaging van de accuraatheid van zo'n 0.002

# Qua topologie van het netwerk heb ik er voor gekozen om één van de hidden layers te verwijderen. Dit maakt het netwerk
# zo'n 2 keer sneller en de accuraatheid bij de verificatieset gaat maar ongeveer 0.01 omlaag.
model = Sequential()
model.add(Dense(512, activation='hard_sigmoid', input_shape=(784,)))
model.add(Dropout(0.2))
#model.add(Dense(512, activation='hard_sigmoid'))
#model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

#Deze functie zorgt ervoor dat het model gebruikt kan worden door Keras en Tensorflow
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])