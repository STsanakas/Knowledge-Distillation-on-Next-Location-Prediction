import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, LSTM, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os, random
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from tflite_utilities import tflite_evaluate, tflite_class_evaluate, tflite_converter
from geolife_data_utilities import get_data_path, get_geolife_path
# from hypertuning import genetic

def transportation_classifier(undersampling=False, train_new=False):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    cols=['bike predicted','bus predicted','car predicted','subway predicted','taxi predicted','train predicted','walk predicted']
    rows=['bike','bus','car','subway','taxi','train','walk']
    datasets=['training', 'testing']
    callback=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, min_delta=0.0001, restore_best_weights=True)]
    X = np.load(get_geolife_path()+'Geolife_datasets/classificationX.npy')
    Y = np.load(get_geolife_path()+'Geolife_datasets/classificationY.npy')

    if undersampling:
        labels=[]
        bike_pos=[]
        bus_pos=[]
        car_pos=[]
        subway_pos=[]
        taxi_pos=[]
        train_pos=[]
        walk_pos=[]
        for i in range(Y.shape[0]):
            labels.append(np.argmax(Y[i]))
        for index, elem in enumerate(np.argmax(Y, axis=1)):
                if elem == 0:
                    bike_pos.append(index)
                elif elem == 1:
                    bus_pos.append(index)
                elif elem == 2:
                    car_pos.append(index)
                elif elem == 3:
                    subway_pos.append(index)
                elif elem == 4:
                    taxi_pos.append(index)
                elif elem == 5:
                    train_pos.append(index)
                elif elem == 6:
                    walk_pos.append(index)
        size = min(len(bike_pos), len(bus_pos), len(car_pos), len(subway_pos), len(taxi_pos), len(train_pos), len(walk_pos))
        bus_data = random.sample(bus_pos, size)
        bike_data = random.sample(bike_pos, size)
        car_data = random.sample(car_pos, size)
        subway_data = random.sample(subway_pos, size)
        taxi_data = random.sample(taxi_pos, size)
        train_data = random.sample(train_pos, size)
        walk_data = random.sample(walk_pos, size)
        newX=[]
        newY=[]
        for index in bus_data:
            newX.append(X[index])
            newY.append(Y[index])
        for index in bike_data:
            newX.append(X[index])
            newY.append(Y[index])
        for index in car_data:
            newX.append(X[index])
            newY.append(Y[index])
        for index in subway_data:
            newX.append(X[index])
            newY.append(Y[index])
        for index in taxi_data:
            newX.append(X[index])
            newY.append(Y[index])
        for index in train_data:
            newX.append(X[index])
            newY.append(Y[index])
        for index in walk_data:
            newX.append(X[index])
            newY.append(Y[index])

        X = np.array(newX)
        Y = np.array(newY)

    X=X*[1,1/360]
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=42)

    traintestdistribution=np.zeros((2,7))
    for i in range(trainX.shape[0]):
        traintestdistribution[0, np.argmax(trainY[i])]+=1
    for i in range(testX.shape[0]):
        traintestdistribution[1, np.argmax(trainY[i])]+=1    
    print(tabulate(traintestdistribution, headers=rows, showindex=datasets))

    if train_new:
        model = Sequential()
        model.add(LSTM(640))
        model.add(Flatten())
        model.add(Dense(200, activation='relu'))  
        model.add(Dropout(0.4))
        model.add(Dense(200, activation='relu'))  
        model.add(Dropout(0.4))
        model.add(Dense(200, activation='sigmoid'))  
        model.add(Dropout(0.4))
        model.add(Dense(7, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(trainX, trainY, epochs=200, batch_size=32, verbose=1, validation_split=0.2, callbacks=callback)
        model.evaluate(testX, testY, verbose=1)
        model.save(get_geolife_path()+'Geolife_models/class_model.h5')

        predictions=np.zeros((7,7))
        a=model.predict(testX)
        for i in range(testX.shape[0]):        
            predictions[np.argmax(testY[i]), np.argmax(a[i])]+=1

        print()
        print(tabulate(traintestdistribution, headers=rows, showindex=datasets))
        print()
        print(tabulate(predictions, headers=cols, showindex=rows))
        tflite_error = tflite_evaluate(tflite_converter(model, get_geolife_path()+'Geolife_models/tf_lite/class_model.tflite'), testX, testY)        

    # tflite_class_evaluate(get_geolife_path()+'Geolife_models/tf_lite/class_model.tflite', testX, testY)
    tflite_class_evaluate(get_geolife_path()+'Geolife_models/tf_lite/class_model.tflite', testX, testY)

transportation_classifier(True, False)