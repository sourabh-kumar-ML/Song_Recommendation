import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,GlobalAveragePooling2D
from keras.layers import Flatten,Dense,Dropout,LSTM,Bidirectional,TimeDistributed
from keras.layers import BatchNormalization,Activation,ConvLSTM2D
from keras.callbacks import ModelCheckpoint,EarlyStopping
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50,InceptionV3,VGG19,Xception
import warnings


class CNNmodel:
    def __init__(self,data_directory,input_shape=None,batch_size=16):
        warnings.filterwarnings(action='ignore')

        #getting data
        self.height,self.width,self.depth = (48,48,1)
        self.data_gen = ImageDataGenerator(validation_split=0.2,rescale=1./255, horizontal_flip=True,)
        self.train_data = self.data_gen.flow_from_directory(directory = data_directory,
                                                target_size=(48,48),
                                                batch_size=batch_size,
                                                class_mode='categorical',
                                                shuffle=True,
                                                color_mode="grayscale",
                                                subset='training')
        self.val_data = self.data_gen.flow_from_directory(directory = data_directory,
                                                target_size=(48,48),
                                                batch_size=1,
                                                class_mode='categorical',
                                                shuffle=True,
                                                color_mode="grayscale",
                                                subset='validation')
        
        #Preparing Model
        self.model = Sequential()
        
        #1st Block
        self.model.add(Conv2D(64,(5,5),input_shape = (self.height,self.width,self.depth),activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, kernel_size=(3, 3),padding="same", activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        
        #Output Layer
        self.model.add(Flatten())
        self.model.add(Dense(512,activation="relu"))
        self.model.add(Dense(4,activation='softmax'))
        
    def summary(self):
        print(self.model.summary())
    
    def train(self,optimizer='adam',loss='categorical_crossentropy',epoch=20):
        print("[INFO] training network...")
        self.model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
        
        es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=4)
        self.history = self.model.fit(self.train_data,epochs=epoch, validation_data=self.val_data,verbose=1,callbacks=[es])
        
        print("[INFO] Saving model...")
        pickle.dump(self.model,open('cnn_model.pkl', 'wb'))
        print("[INFO] Done...")
        
class Vgg19(CNNmodel):
    def __init__(self,data_directory,batch_size=16):
        self.cnn = CNNmodel(data_directory)
        self.height,self.width,self.depth = (48,48,1)
        self.batch_size=batch_size
        self.vgg = VGG19(include_top=True, weights=None,input_shape =(self.height,self.width,self.depth), classes = 4)
        
    def summary(self):
        print(self.vgg.summary())
        
    def train(self,optimizer='adam',loss='categorical_crossentropy',epoch=20):
        
        self.vgg.compile(loss="categorical_crossentropy", optimizer=optimizer,metrics=["accuracy"])
        print("[INFO] training network...")
        es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=4)
        self.history = self.vgg.fit(self.cnn.train_data,epochs=epoch, validation_data=self.cnn.val_data,verbose=1,callbacks=[es])
        
        print("[INFO] Saving model...")
        pickle.dump(self.vgg,open('vgg_model.pkl', 'wb'))
        print("[INFO] Done...")
        
class Resnet(CNNmodel):
    def __init__(self,data_directory,batch_size=16):
        self.cnn = CNNmodel(data_directory)
        self.height,self.width,self.depth = (48,48,1)
        self.batch_size=batch_size
        self.resnet = ResNet50(include_top=True, weights=None,input_shape =(self.height,self.width,self.depth), classes = 4)
        
    def summary(self):
        print(self.resnet.summary())
        
    def train(self,optimizer='adam',loss='categorical_crossentropy',epoch=20):
        
        self.resnet.compile(loss="categorical_crossentropy", optimizer=optimizer,metrics=["accuracy"])
        print("[INFO] training network...")
        es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=4)
        self.history = self.resnet.fit(self.cnn.train_data,epochs=epoch, validation_data=self.cnn.val_data,verbose=1,callbacks=[es])
        
        print("[INFO] Saving model...")
        pickle.dump(self.resnet,open('resnet.pkl', 'wb'))
        print("[INFO] Done...")
        

class Xceptionmodel(CNNmodel):
    def __init__(self,data_directory,batch_size=16):
        self.cnn = CNNmodel(data_directory)
        self.height,self.width,self.depth = (48,48,1)
        self.batch_size=batch_size
        self.xception = Xception(include_top=True, weights=None,input_shape =(self.height,self.width,self.depth), classes = 4)
        
    def summary(self):
        print(self.xception.summary())
        
    def train(self,optimizer='adam',loss='categorical_crossentropy',epoch=20):
        
        self.xception.compile(loss="categorical_crossentropy", optimizer=optimizer,metrics=["accuracy"])
        print("[INFO] training network...")
        es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=4)
        self.history = self.xception.fit(self.cnn.train_data,epochs=epoch, validation_data=self.cnn.val_data,verbose=1,callbacks=[es])
        
        print("[INFO] Saving model...")
        pickle.dump(self.xception,open('xception.pkl', 'wb'))
        print("[INFO] Done...")
        
        
class Conv_Lstm():
    def __init__(self,input_shape=None):
        warnings.filterwarnings(action="ignore")
        self.model=Sequential()
        self.model.add(TimeDistributed(Conv2D(64,(5,5),input_shape=(1,48,48,1),activation='relu')))
        self.model.add(TimeDistributed(BatchNormalization()))
        self.model.add(TimeDistributed(Conv2D(64, kernel_size=(3, 3), activation='relu')))
        self.model.add(TimeDistributed(BatchNormalization()))
        self.model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        self.model.add(TimeDistributed(Dropout(0.25)))

        self.model.add(TimeDistributed(Conv2D(128, kernel_size=(3, 3),padding="same", activation='relu')))
        self.model.add(TimeDistributed(BatchNormalization()))
        self.model.add(TimeDistributed(Conv2D(256, kernel_size=(3, 3),padding="same", activation='relu')))
        self.model.add(TimeDistributed(BatchNormalization()))
        self.model.add(TimeDistributed(Conv2D(256, kernel_size=(3, 3),padding="same", activation='relu')))
        self.model.add(TimeDistributed(BatchNormalization()))
        self.model.add(TimeDistributed(Conv2D(128, kernel_size=(3, 3),padding="same", activation='relu')))
        self.model.add(TimeDistributed(BatchNormalization()))
        self.model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        self.model.add(TimeDistributed(Dropout(0.25)))

        self.model.add(TimeDistributed(Conv2D(256, kernel_size=(3, 3),padding="same", activation='relu')))
        self.model.add(TimeDistributed(BatchNormalization()))
        self.model.add(TimeDistributed(Conv2D(512, kernel_size=(3, 3),padding="same", activation='relu')))
        self.model.add(TimeDistributed(BatchNormalization()))
        self.model.add(TimeDistributed(Conv2D(512, kernel_size=(3, 3),padding="same", activation='relu')))
        self.model.add(TimeDistributed(BatchNormalization()))
        self.model.add(TimeDistributed(Conv2D(128, kernel_size=(3, 3),padding="same", activation='relu')))
        self.model.add(TimeDistributed(BatchNormalization()))
        self.model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        self.model.add(TimeDistributed(Dropout(0.25)))

        self.model.add(TimeDistributed(Conv2D(256, kernel_size=(3, 3),padding="same", activation='relu')))
        self.model.add(TimeDistributed(BatchNormalization()))
        self.model.add(TimeDistributed(Conv2D(256, kernel_size=(3, 3),padding="same", activation='relu')))
        self.model.add(TimeDistributed(BatchNormalization()))
        self.model.add(TimeDistributed(Conv2D(256, kernel_size=(3, 3),padding="same", activation='relu')))
        self.model.add(TimeDistributed(BatchNormalization()))
        self.model.add(TimeDistributed(Conv2D(256, kernel_size=(3, 3),padding="same", activation='relu')))
        self.model.add(TimeDistributed(BatchNormalization()))
        self.model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        self.model.add(TimeDistributed(Dropout(0.25)))

        self.model.add(TimeDistributed(Flatten()))
        self.model.add(LSTM(64,return_sequences=True,activation="relu"))
        self.model.add(LSTM(128,return_sequences=True,activation="relu"))
        self.model.add(Bidirectional(LSTM(64,activation="relu")))
        self.model.add(Dense(1024,activation="relu"))
        self.model.add(Dense(4,activation='softmax'))

    
    def train(self,X_train,Y_train,X_test,Y_test,optimizer='adam',loss='categorical_crossentropy',epoch=20,batch_size=16):
        print("[INFO] training network...")
        self.model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
        
        es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=5)
        self.history = self.model.fit(X_train,Y_train,epochs=epoch, validation_data=(X_test,Y_test),batch_size=batch_size,verbose=1,callbacks=[es])
        
        print("[INFO] Saving model...")
        pickle.dump(self.model,open('convLstm_model.pkl', 'wb'))
        print("[INFO] Done...")