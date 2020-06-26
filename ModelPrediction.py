
import os
import cv2
import validators
from tensorflow.keras import models, backend, layers

import numpy as np

class_names = ['Bentley Continental GT Coupe 2007', 'Chrysler 300 SRT-8 2010',
 'GMC Savana Van 2012', 'Jaguar XK XKR 2012', 'Mitsubishi Lancer Sedan 2012']

IMG_HEIGHT = 300
IMG_WIDTH = 300
COLOUR_CHANNELS = 3

model = models.load_model('../Models/convolutionalTransfer.h5')

def PredictImage(source):
    if os.path.exists(source):
        raw_image = cv2.imread(source)
        print("Got Image")
    else:
        try:
            print("Cant find image")
            file_to_delete = source.split("/")[-1]
            urllib.request.urlretrieve(source, file_to_delete)
            raw_image = cv2.imread(file_to_delete)
            os.remove(file_to_delete)
        except Exception:
            return "Input is neither a valid file nor url"

    input_shape = (IMG_HEIGHT, IMG_WIDTH, COLOUR_CHANNELS)
    scaled_image = cv2.resize(raw_image, dsize=(IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite("scaled.jpg", scaled_image)

    scaled_image = scaled_image/255
    image_array = scaled_image.reshape(1, IMG_HEIGHT, IMG_WIDTH, COLOUR_CHANNELS)

    predictions = model.predict(image_array)
    print("It is a: " + class_names[np.argmax(predictions)])

PredictImage('C:\\Users\\Sam Milward\\Documents\\Third Year\\AI\\stanfordcarsfcs\\ReducedDataset\\Validation\\Bentley Continental GT Coupe 2007\\00171.jpg')
PredictImage('C:\\Users\\Sam Milward\\Documents\\Third Year\\AI\\stanfordcarsfcs\\ReducedDataset\\Validation\\Chrysler 300 SRT-8 2010\\01089.jpg')
PredictImage('C:\\Users\\Sam Milward\\Documents\\Third Year\\AI\\stanfordcarsfcs\\ReducedDataset\\Validation\\GMC Savana Van 2012\\06775.jpg')