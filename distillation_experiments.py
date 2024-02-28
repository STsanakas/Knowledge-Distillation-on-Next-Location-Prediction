import os
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
import math
import keras.backend as K
import tensorflow as tf
from keras.layers import Dense, Flatten, LSTM
from time import time
from Distiller import Distiller
from mobility import cartesian_error
from tflite_utilities import tflite_converter, tflite_evaluate
from geolife_data_utilities import get_data, get_geolife_path

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# print(tf.config.list_physical_devices('GPU'))

data_path = get_geolife_path()


def create_student():
    network = Sequential()
    network.add(LSTM(15, input_shape=(15, 2)))
    network.add(Flatten())
    network.add(Dense(20, activation='relu'))
    network.add(Dense(2))
    network.compile(loss=cartesian_error, optimizer='Adam')
    return network


def distill_on_student(teacher_file, student, mode):
    X = np.delete(np.load(data_path+'Geolife_datasets/'+mode+'X.npy'), 0, 0)
    Y = np.delete(np.load(data_path+'Geolife_datasets/'+mode+'Y.npy'), 0, 0)
    X = X*[1, 1/360]
    Y = Y*[1, 1/360]
    if mode == 'dataset':
        X = X[:50000]
        Y = Y[:50000]
    trainX, testX, trainY, testY = train_test_split(
        X, Y, test_size=0.33, random_state=42)
    # student = load_model(data_path+'Geolife_models/'+student_file+'.h5', custom_objects={'cartesian_error': cartesian_error})
    teacher = load_model(data_path+'Geolife_models/'+teacher_file +
                         '.h5', custom_objects={'cartesian_error': cartesian_error})
    bsize = 32
    trsize = bsize*int(trainX.shape[0]/bsize)
    tssize = bsize*int(testX.shape[0]/bsize)
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=[],
        student_loss_fn=cartesian_error,
        distillation_loss_fn=keras.losses.MeanSquaredError(),
        alpha=0.1,
        temperature=10,
    )
    # Distill teacher to student
    start = time()
    distiller.fit(trainX[:trsize], trainY[:trsize], epochs=20, verbose=0)
    # Evaluate student on test dataset
    error = distiller.evaluate(testX[:tssize], testY[:tssize], verbose=0)
    # print(mode, 'distillation error:', np.average(error))
    # print(mode, 'distillation time:', int(time()-start))
    return distiller.student, np.average(error), int(time()-start), testX.shape[0]


def train_student(student, mode):
    trainX, testX, trainY, testY = get_data(mode)
    start = time()
    history = student.fit(trainX, trainY, epochs=20,
                          batch_size=32, verbose=0, validation_split=0.2)
    error = student.evaluate(testX, testY, verbose=0)
    # print(mode, 'student training error:', round(error,2))
    # print(mode, 'student training time:', int(time()-start))
    student.save(data_path+'Geolife_models/'+mode+'.h5')
    return student, error, int(time()-start), testX.shape[0]


def evaluate_on(model, mode):
    trainX, testX, trainY, testY = get_data(mode)
    network = load_model(data_path+'Geolife_models/'+model+'.h5',
                         custom_objects={'cartesian_error': cartesian_error})
    error = network.evaluate(testX, testY, verbose=0)
    # print(mode, 'teacher error:', round(error,2))
    return error, testX.shape[0]

def clear_nans(error, size, mode):
    if error > 4000 or math.isnan(error):
        print('REMOVE', size, 'OBJECTS OF', error, 'ERROR FOR', mode)
        return 0, 0
    return error, size

def run_experiments(filename, teacher_model_name):
    means = ['bike', 'bus', 'car', 'subway', 'taxi', 'train', 'walk']
    results = np.zeros((len(means), 6))
    sizes = np.zeros((len(means)))
    map = ['NW', 'NE', 'SW', 'SE']
    create_student().save(data_path+'Geolife_models/student.h5')
    
    with open(data_path+filename+'.csv', 'w') as fd:
        fd.write('')
    for area in map:
        with open(data_path+filename+'.csv', 'a') as fd:
            fd.write(','+area+'\n')
            # fd.write(',teacher model,student self teaching,train time, student distilled, distillation time, data'+'\n')
            fd.write(',teacher model, student self-teaching, student distilled, student double-distilled, student fine-tuned, tf-lite conversion'+'\n')
        for i, mean in enumerate(means):
            mode = area+'_'+mean
            old_model, size = evaluate_on(teacher_model_name, mode)
            old_model, size = clear_nans(old_model, size, mode)
            results[i, 0] += round(old_model*size, 2)
            _, train_error, train_time, size = train_student(create_student(), mode)
            train_error, size = clear_nans(train_error, size, mode)
            results[i, 1] += round(train_error*size, 2)
            _, distillation_error, distillation_time, size = distill_on_student(teacher_model_name, create_student(), mode)
            distillation_error, size = clear_nans(distillation_error, size, mode)
            results[i, 2] += round(distillation_error*size, 2)
            mystudent = load_model(data_path+'Geolife_models/student.h5',custom_objects={'cartesian_error': cartesian_error})
            _, doubledistillation_error, doubledistillation_time, size = distill_on_student(teacher_model_name, mystudent, mode)
            doubledistillation_error, size = clear_nans(doubledistillation_error, size, mode)
            results[i, 3] += round(doubledistillation_error*size, 2)
            mystudent2 = load_model(data_path+'Geolife_models/student.h5', custom_objects={'cartesian_error': cartesian_error})
            finetuned, finetune_error, finetune_time, size = train_student(mystudent2, mode)
            finetune_error, size = clear_nans(finetune_error, size, mode)
            results[i, 4] += round(finetune_error*size, 2)
            _, testX, _, testY = get_data(mode)
            size = testX.shape[0]
            tflite_error = tflite_evaluate(tflite_converter(finetuned, data_path+'Geolife_models/tf_lite/'+mode+'.tflite'), testX, testY)
            tflite_error, size = clear_nans(tflite_error, size, mode)
            results[i, 5] += round(tflite_error*size, 2)
            sizes[i] += size
            

            with open(data_path+filename+'.csv','a') as fd:
                fd.write(mean+','+str(round(old_model, 2))+','+str(round(train_error,2))+','+str(round(distillation_error,2))+','+str(round(doubledistillation_error,2))+','+str(round(finetune_error,2))+','+str(round(tflite_error,2))+'\n')        
        print(results)
        with open(data_path+filename+'.csv','a') as fd:
            fd.write(''+'\n')
    results = results/sizes[:,None]
    np.savetxt(data_path+"condensed_results.csv", results, delimiter=',', header="teacher model,student self-teaching,student distilled,student double-distilled,student fine-tuned, tf lite", fmt='%1.2f')
    print(results)
    print(sizes)

run_experiments('experiment_outputs', 'Unlabeled')
