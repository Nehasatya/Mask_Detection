from cgi import test
from msilib.schema import Directory
import warnings
warnings.filterwarnings('ignore')
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from keras.layers import Conv2D,Dense,MaxPooling2D,Activation,Dropout,Flatten,AveragePooling2D
from tensorflow.keras.optimizers import Adam
#creating two list ine to store img data and one to store result i.e.,,(with_mask,without_mask)
data=[]
labels=[]
Directory=r"G:\dataset"
Categories=["with_mask","without_mask"]
for category in Categories:
    path=os.path.join(Directory,category)
    for img in os.listdir(path):
        #For getting img path
        img_path=os.path.join(path,img)
        #reading img
        img=cv.imread(img_path)
        #resizing img
        img=cv.resize(img,(32,32))
        #appending with the list data
        data.append(img)
        #appending result such as "with_mask","without_mask" into labels
        labels.append(category)
lb=LabelBinarizer()
#assign 0 for with_mask and 1 for without_mask    [0]
labels = lb.fit_transform(labels)
#converts that into matrix  [1 0 0 0]
labels = to_categorical(labels)
data=np.array(data,dtype="float32")
labels=np.array(labels)
#data splitting
(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=0.2,stratify=labels,random_state=42)
#data preprocessing
def preprocessing(img):
    #to convert float point numbers to int 
    #img shape :(32,32,3)
    img=img.astype(np.uint8)
    #to convert rgb to Gray shape
    #img shape :(32,32)
    img=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img=cv.equalizeHist(img)
    #dividing with 255 to normalize values betwwen 0 and 1
    img=img/255
    return img
#trainX shape:(3066,32,32)
trainX=np.array(list(map(preprocessing, trainX)))
#testX shape:(767,32,32,3)
testX=np.array(list(map(preprocessing, testX)))
trainX=trainX.reshape(trainX.shape[0], trainX.shape[1], trainX.shape[2],1)
#trainX shape:(3066,32,32,1)
testX=testX.reshape(testX.shape[0], testX.shape[1], testX.shape[2],1)
#testX shape:(767,32,32,1)
print(trainX)
print(testX)
#data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")
aug.fit(trainX)
#building model
def myModel():
    sizeOfFilter1=(3,3)
    sizeOfFilter2=(3,3)
    sizeOfPool=(2,2)
    model=Sequential()
    # print(trainX.shape)
    # print(testX.shape)
    model.add((Conv2D(32, sizeOfFilter1, input_shape=(32,32,1),activation='relu')))
    model.add((Conv2D(32, sizeOfFilter1,activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model
model=myModel()
#to display model information
print(model.summary())
#training model
history=model.fit_generator(aug.flow(trainX, trainY,batch_size=32),
	steps_per_epoch=len(trainX)//32,
	epochs=20,
	validation_data=(testX,testY),
	shuffle=1)
#saving model
model.save("MaskModel.h5")



