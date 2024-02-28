import os
import tensorflow as tf
from mobility import cartesian_eval, cartesian_error
from tensorflow import lite
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size

def convert_bytes(size, unit=None):
    if unit == "KB":
        return print('File size: ' + str(round(size / 1024, 3)) + ' Kilobytes')
    elif unit == "MB":
        return print('File size: ' + str(round(size / (1024 * 1024), 3)) + ' Megabytes')
    else:
        return print('File size: ' + str(size) + ' bytes')


def tflite_converter(from_model, path):
    converter = lite.TFLiteConverter.from_keras_model(from_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter=True
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
    tfmodel = converter.convert()
    open(path, 'wb').write(tfmodel)
    return path

def tflite_infer(model, input):
    interpreter = tf.lite.Interpreter(model_path = model)
    input_details = interpreter.get_input_details()    
    output_details = interpreter.get_output_details()
    myshape = input_details[0]['shape']
    myshape[0]=input.shape[0]
    interpreter.resize_tensor_input(input_details[0]['index'], myshape)    
    interpreter.allocate_tensors()
    input32 = np.array(input, dtype=input_details[0]['dtype'])

    interpreter.set_tensor(input_details[0]['index'], input32)
    interpreter.invoke()
    tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
    return tflite_model_predictions

def tflite_evaluate(model, testX, testY):
    interpreter = tf.lite.Interpreter(model_path = model)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.resize_tensor_input(input_details[0]['index'], testX.shape)
    interpreter.resize_tensor_input(output_details[0]['index'], testY.shape)    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.allocate_tensors()
    testX32 = np.array(testX, dtype=input_details[0]['dtype'])

    interpreter.set_tensor(input_details[0]['index'], testX32)
    interpreter.invoke()
    tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
    print(model.split('/')[-1].split('.')[0], 'tensorflow lite error:', round(cartesian_eval(testY, tflite_model_predictions),2))
    return cartesian_eval(testY, tflite_model_predictions)


def tflite_class_evaluate(model, testX, testY):
    interpreter = tf.lite.Interpreter(model_path = model)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.resize_tensor_input(input_details[0]['index'], testX.shape)
    interpreter.resize_tensor_input(output_details[0]['index'], testY.shape)    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.allocate_tensors()
    testX32 = np.array(testX, dtype=input_details[0]['dtype'])

    interpreter.set_tensor(input_details[0]['index'], testX32)
    interpreter.invoke()
    tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
    print(tflite_model_predictions.shape)
    from sklearn.metrics import accuracy_score
    tflite_model_predictions = (tflite_model_predictions > 0.5) 
    print(model.split('/')[-1].split('.')[0], 'tensorflow lite accuracy:', accuracy_score(testY, tflite_model_predictions),2)
    return cartesian_eval(testY, tflite_model_predictions)



def tf_infer(model, input):  
  network = tf.keras.models.load_model(model, custom_objects={'cartesian_error': cartesian_error})
  pred = network.predict(input)
  return pred

