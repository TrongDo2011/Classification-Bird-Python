import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
def createMapping(directory):
    Name=[]
    for file in os.listdir(directory):
        Name+=[file]
    print(Name)
    print(len(Name))

    N=[]
    for i in range(len(Name)):
        N+=[i]

    mapping=dict(zip(Name,N))
    reverse_mapping=dict(zip(N,Name))
    return  reverse_mapping

def mapper(value,reverse_mapping):
    return reverse_mapping[value]

def split_data():
    dataset=[]
    testset=[]
    count=0
    for file in os.listdir(directory):
        path=os.path.join(directory, file)
        t=0
        for im in os.listdir(path):
            image=load_img(os.path.join(path, im), grayscale=False, color_mode='rgb', target_size=(180, 180))
            image=img_to_array(image)
            image=image/255.0
            if t<=31:
                dataset+=[[image,count]]
            else:
                testset+=[[image,count]]
            t+=1
        count=count+1
    data,labels0=zip(*dataset)
    test,testlabels0=zip(*testset)
    labels1=to_categorical(labels0)
    labels=np.array(labels1)
    data=np.array(data)
    test=np.array(test)
    return data, test, labels, labels1, testlabels0, labels0
def split_data2(data, labels):
    trainx, testx, trainy, testy = train_test_split(data, labels, test_size=0.25, random_state= 44)
    print(trainx.shape)
    print(testx.shape)
    print(trainy.shape)
    print(testy.shape)
    return trainx, testx, trainy, testy

def Train(trainx,trainy,testx,testy):
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=20, zoom_range=0.2,
                                 width_shift_range=0.2, height_shift_range=0.2, shear_range=0.1, fill_mode="nearest")
    pretrained_model3 = tf.keras.applications.DenseNet201(input_shape=(180, 180, 3),
                                                          include_top=False, weights='imagenet',
                                                          pooling='avg')
    pretrained_model3.trainable = False
    inputs3 = pretrained_model3.input
    x3 = tf.keras.layers.Dense(128, activation='relu')(pretrained_model3.output)
    outputs3 = tf.keras.layers.Dense(200, activation='softmax')(x3)
    model = tf.keras.Model(inputs=inputs3, outputs=outputs3)
    model.compile(optimizer= Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    his = model.fit(datagen.flow(trainx, trainy, batch_size=128), validation_data=(testx, testy), epochs= 200, steps_per_epoch=trainx.shape[0]/128)
    return his, model
def printclassification_report(testx, testy, model):
    y_pred= model.predict(testx)
    pred=np.argmax(y_pred,axis=1)
    ground = np.argmax(testy,axis=1)
    print(classification_report(ground,pred))

def Values(his):
    get_acc = his.history['accuracy']
    value_acc = his.history['val_accuracy']
    get_loss = his.history['loss']
    validation_loss = his.history['val_loss']
    return get_acc, value_acc, get_loss, validation_loss

def paintAcc(get_acc,value_acc):
    epochs = range(len(get_acc))
    plt.plot(epochs, get_acc, 'r', label='Accuracy of Training data')
    plt.plot(epochs, value_acc, 'b', label='Accuracy of Validation data')
    plt.title('Training vs validation accuracy')
    plt.legend(loc=0)
    plt.figure()
    plt.show()

def paintLoss(get_loss,validation_loss):
    epochs = range(len(get_loss))
    plt.plot(epochs, get_loss, 'r', label='Loss of Training data')
    plt.plot(epochs, validation_loss, 'b', label='Loss of Validation data')
    plt.title('Training vs validation loss')
    plt.legend(loc=0)
    plt.figure()
    plt.show()

def TestImage(Linkfile,model,mapper):
    image=load_img(Linkfile,target_size=(180, 180))
    image=img_to_array(image)
    image=image/255.0
    prediction_image=np.array(image)
    prediction_image= np.expand_dims(image, axis=0)
    prediction=model.predict(prediction_image)
    value=np.argmax(prediction)
    move_name=mapper(value)
    print("Prediction is {}.".format(move_name))

def accuracy(test,model,testlabels0):
    print(test.shape)
    pred2=model.predict(test)
    print(pred2.shape)
    PRED=[]
    for item in pred2:
        value2=np.argmax(item)
        PRED+=[value2]
    ANS=testlabels0
    accuracy=accuracy_score(ANS,PRED)
    print(accuracy)

def saveModel(model):
    model.save("Mode_Tree_Dense_200_ver1.h5")
    print("Mode_Tree_Dense_200_ver1.h5")

if __name__ == '__main__':
    directory = 'DataImage'
    createMapping(directory)
    data, test, labels, labels1, testlabels0, labels0 = split_data()
    trainx, testx, trainy, testy = split_data2(data, labels)
    his, model = Train(trainx,trainy,testx,testy)
    printclassification_report(testx, testy, model)
    get_acc, value_acc, get_loss, validation_loss = Values(his)
    paintAcc(get_acc, value_acc)
    paintLoss(get_loss, validation_loss)
    accuracy(test,model,testlabels0)
    saveModel(model)
