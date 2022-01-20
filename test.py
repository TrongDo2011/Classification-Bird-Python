import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
directory = 'images'
Name=[]
for file in os.listdir(directory):
    Name+=[file]
print(Name)
print(len(Name))
N = []
for i in range(len(Name)):
    N += [i]

mapping = dict(zip(Name, N))
reverse_mapping = dict(zip(N, Name))

print(mapping)
def mapper(value):
    return reverse_mapping[value]

model = load_model('Mode_Bird_Dense_test_ver1.h5')

image=load_img("test/Crested_Auklet_test3.jpg",target_size=(180,180))
image=img_to_array(image)
image=image/255.0
prediction_image=np.array(image)
prediction_image= np.expand_dims(image, axis=0)

prediction=model.predict(prediction_image)
value=np.argmax(prediction)
move_name=mapper(value)
print("Prediction is {}.".format(move_name))

imageshow = mpimg.imread("test/Crested_Auklet_test3.jpg")
plt.imshow(imageshow)
plt.title("Prediction is {}.".format(move_name))
plt.show()