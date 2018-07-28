import cv2
import numpy as np
from tensorflow import keras
import os
import matplotlib.pyplot as plt


# Processes input images
folder = "prediction_inputs"
images = []
processed_images = []
for file in os.listdir(folder):
    image = cv2.imread(os.path.join(folder, file))
    rescaled = cv2.resize(image, (256, 256))/255.0
    if image is not None:
        images.append(image)
        processed_images.append(rescaled)

processed_images = np.array(processed_images)


# Loads pre-trained model
model = keras.models.load_model('Dogs-vs-Cats_model.h5')


# Feeds input data to model
predictions = model.predict(processed_images)


# Displays image and prediction
plt.ion()
plt.rcParams['toolbar'] = 'None'
plt.rcParams['font.size'] = 18
plt.rcParams['figure.figsize'] = [8.0, 6.0]

for index in range(len(images)):
    plt.figure("Dog vs Cat Classifier")

    plt.xticks([])
    plt.yticks([])

    if predictions[index][0] > 0.5:
        title = "Prediction: Dog \n"
        x_label = "\n Confidence: {:5.2f}%".format(200*(predictions[index][0]-0.5))
    else:
        title = "Prediction: Cat \n"
        x_label = "\n Confidence: {:5.2f}%".format(200*(0.5-predictions[index][0]))
    plt.title(title)
    plt.xlabel(x_label)

    plt.imshow(cv2.cvtColor(images[index], cv2.COLOR_BGR2RGB))

    plt.show()
    if plt.waitforbuttonpress():
        plt.close()

