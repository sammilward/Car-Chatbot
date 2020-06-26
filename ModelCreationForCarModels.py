import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

batch_size = 6
epochs = 100
IMG_HEIGHT = 300
IMG_WIDTH = 300

train_dir = "C:\\Users\\Sam Milward\\Documents\\Third Year\\AI\\stanfordcarsfcs\\ReducedDataset2\\Train"
valid_dir = "C:\\Users\\Sam Milward\\Documents\\Third Year\\AI\\stanfordcarsfcs\\ReducedDataset2\\Valdidation"

train_image_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=35, zoom_range=0.5, width_shift_range=.15, height_shift_range=.15)
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')

validation_image_generator = ImageDataGenerator(rescale=1./255)

validation_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=valid_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')

base_model=MobileNet(weights='imagenet',include_top=False)

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dense(512,activation='relu')(x)
x=Dense(256,activation='relu')(x)
preds=Dense(5,activation='softmax')(x)

model=Model(inputs=base_model.input,outputs=preds)

for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

#with tf.device('/device:GPU:0'):
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch = train_data_gen.samples // batch_size,
    epochs = epochs,
    validation_data = validation_data_gen,
    validation_steps = validation_data_gen.samples // batch_size
)

model.save('TransferLearningModelForCarModels.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()