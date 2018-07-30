from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Pre-processing image data
data_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)


# Generating batches of validation data
data_dir = 'data'

img_width = 256
img_height = 256

validation_generator = data_gen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=50,
    class_mode='binary',
    subset="validation"
)


# Loads pre-trained model
model = keras.models.load_model('Dogs-vs-Cats_model.h5')


# Evaluates accuracy of pre-trained model
test_loss, test_acc = model.evaluate_generator(validation_generator, steps=100, verbose=1)

print('Test accuracy: {:5.2f}%'.format(100 * test_acc))
print('Test loss: ',  test_loss)
