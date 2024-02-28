# from tabnanny import verbose
from tensorflow.keras.models import load_model
from mobility import cartesian_error, cartesian_eval
from geolife_data_utilities import get_data, get_geolife_path
from tflite_utilities import tflite_infer
from tqdm import tqdm
from tabulate import tabulate
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def single(X, Y):
    return X.reshape(1,X.shape[0],X.shape[1]), Y.reshape(1,Y.shape[0])

def agnostic_prediction(input, output, classification, bike_model, bus_model, car_model, subway_model, taxi_model, train_model, walk_model):
    class_pred = classification.predict(input, verbose=0)
    pred_means = np.argmax(class_pred)    
    if pred_means==0:
        eval = cartesian_eval(output, bike_model.predict(input, verbose=0))                
    elif pred_means==1:
        eval = cartesian_eval(output, bus_model.predict(input, verbose=0))        
    elif pred_means==2:
        eval = cartesian_eval(output, car_model.predict(input, verbose=0))        
    elif pred_means==3:
        eval = cartesian_eval(output, subway_model.predict(input, verbose=0))        
    elif pred_means==4:
        eval = cartesian_eval(output, taxi_model.predict(input, verbose=0))        
    elif pred_means==5:
        eval = cartesian_eval(output, train_model.predict(input, verbose=0))        
    elif pred_means==6:
        eval = cartesian_eval(output, walk_model.predict(input, verbose=0))        
    return pred_means, eval

def multiple_agnostic(X, Y, areamodel):
    meanss=[]
    errors=[]
    classification = load_model(get_geolife_path()+'Geolife_models/class_model.h5')
    bike_model = load_model(models_path+areamodel+'_bike.h5', custom_objects={'cartesian_error': cartesian_error})
    bus_model = load_model(models_path+areamodel+'_bus.h5', custom_objects={'cartesian_error': cartesian_error})
    car_model = load_model(models_path+areamodel+'_car.h5', custom_objects={'cartesian_error': cartesian_error})
    subway_model = load_model(models_path+areamodel+'_subway.h5', custom_objects={'cartesian_error': cartesian_error})
    taxi_model = load_model(models_path+areamodel+'_taxi.h5', custom_objects={'cartesian_error': cartesian_error})
    train_model = load_model(models_path+areamodel+'_train.h5', custom_objects={'cartesian_error': cartesian_error})
    walk_model = load_model(models_path+areamodel+'_walk.h5', custom_objects={'cartesian_error': cartesian_error})
    for i in tqdm(range(X.shape[0])):
        input, output=single(X[i], Y[i])
        means, error = agnostic_prediction(input, output, classification, bike_model, bus_model, car_model, subway_model, taxi_model, train_model, walk_model)
        meanss.append(means)
        errors.append(error)        
    return errors, meanss

def agnostic_lite(input):
    if input.shape!=(1,15,2):
        input = input.reshape((1,15,2))
    tflite_models_path = get_geolife_path()+'Geolife_models/tf_lite/'    
    class_pred = tflite_infer(tflite_models_path+'class_model.tflite', input)
    pred_means = np.argmax(class_pred)    
    if pred_means==0:
        mymodel = tflite_models_path+'NW_bike.tflite'
    elif pred_means==1:
        mymodel = tflite_models_path+'SE_bus.tflite'                
    elif pred_means==2:
        mymodel = tflite_models_path+'SE_car.tflite'
    elif pred_means==3:
        mymodel = tflite_models_path+'NE_subway.tflite'
    elif pred_means==4:
        mymodel = tflite_models_path+'NW_taxi.tflite'
    elif pred_means==5:
        mymodel = tflite_models_path+'SW_train.tflite'
    elif pred_means==6:
        mymodel = tflite_models_path+'SW_walk.tflite'

    answer = tflite_infer(mymodel, input)
    return pred_means, answer

def eval_aglite(input, output):
    transport, pred = agnostic_lite(input)
    return transport, cartesian_eval(output.reshape(1,2), pred)

data_path = get_geolife_path()+'Geolife/'
models_path = data_path+'Geolife_models/'

def evaluate_unlabeled():
    _, testX, _, testY = get_data('dataset')
    X = testX[100000:]
    Y = testY[100000:]
    means = ['bike', 'bus', 'car', 'subway', 'taxi', 'train', 'walk']
    results = np.zeros((len(means), 2))    
    print(X.shape, Y.shape)
    for i in tqdm(range(X.shape[0])):
        tran, error = eval_aglite(X[i], Y[i])
        results[tran, 0]+=1
        results[tran, 1]+=error
    results[:,1]=np.round(results[:,1]/results[:,0],2)    
    print(tabulate(results, headers=['means of \ntransportation', 'predicted \nin class', 'error(m)'], showindex=means))
    return        

