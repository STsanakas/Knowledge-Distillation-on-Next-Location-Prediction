from tflite_utilities import tflite_infer, tf_infer
import time
import numpy as np
from geolife_data_utilities import get_geolife_path

def keep_time(func, *args):
    start_time=time.time()
    func(*args)
    return(time.time()-start_time)


def implementation_cost(repeats=11, batch=1):
    X1 = np.random.rand(batch,15,2)
    for i in range(repeats):
        print(keep_time(tflite_infer, get_geolife_path()+'Geolife_models/tf_lite/NE_taxi.tflite', X1))
    for i in range(repeats):
	    print(keep_time(tf_infer, get_geolife_path()+'Geolife_models/NE_taxi.h5', X1))
