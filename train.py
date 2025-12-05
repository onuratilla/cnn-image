import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import tensorflow as tf

USE_CIFAR10 = True
BATCH_SIZE = 64
EPOCHS = 40
IMG_SIZE = (32, 32)
NUM_CLASSES = 10

if USE_CIFAR10:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test_cat = to_categorical(y_test, NUM_CLASSES)
    train_datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    train_gen = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
    val_datagen = ImageDataGenerator()
    val_gen = val_datagen.flow(x_test, y_test_cat, batch_size=BATCH_SIZE, shuffle=False)
else:
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_directory('data/train', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
    val_gen = val_datagen.flow_from_directory('data/val', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)
    NUM_CLASSES = train_gen.num_classes

def build_simple_cnn(input_shape=(32,32,3), num_classes=10):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

model = build_simple_cnn((IMG_SIZE[0], IMG_SIZE[1], 3), NUM_CLASSES)
model.compile(optimizer=optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

cb = [
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    callbacks.ModelCheckpoint('models/best_model.h5', save_best_only=True, monitor='val_loss')
]

history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=cb)

model.save('models/final_model.h5')

def plot_history(h):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(h.history['loss'], label='train loss')
    plt.plot(h.history['val_loss'], label='val loss')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(h.history['accuracy'], label='train acc')
    plt.plot(h.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.title('Accuracy')
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/training_curves.png', dpi=150)
    plt.close()

plot_history(history)

y_pred_probs = model.predict(val_gen)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_gen.classes if not USE_CIFAR10 else np.argmax(y_test_cat, axis=1)

cm = confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fname = 'results/cm_norm.png' if normalize else 'results/cm.png'
    plt.savefig(fname, dpi=150)
    plt.close()

class_names = ['class'+str(i) for i in range(NUM_CLASSES)]
plot_confusion_matrix(cm, class_names, False)
plot_confusion_matrix(cm, class_names, True)
