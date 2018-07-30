from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Pre-processing image data
data_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)


# Generating batches of data
data_dir = 'data'

img_width = 256
img_height = 256

train_generator = data_gen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=50,
    class_mode='binary',
    subset="training"
)

validation_generator = data_gen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=50,
    class_mode='binary',
    subset="validation"
)


# Model architecture definition
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation('sigmoid'))


# Compiling model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# Defining callbacks
checkpoint_path = "training_checkpoint/cp.ckpt"

checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_loss',
    verbose=2,
    save_best_only=True,
    mode='min',
    save_weights_only=True
)

early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=5,
    mode='min'
)


# Loading pre-trained weights
model.load_weights(checkpoint_path)


# Training model
model.fit_generator(
    train_generator,
    steps_per_epoch=400,
    epochs=20,
    callbacks=[early_stop, checkpoint],
    validation_data=validation_generator,
    validation_steps=100
)


# Saving trained model
model.save('Dogs-vs-Cats_model.h5')
