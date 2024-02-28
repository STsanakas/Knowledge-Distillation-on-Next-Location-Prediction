from tqdm import tqdm
from geolife_data_utilities import get_geolife_path
import os
import numpy as np

thepath = get_geolife_path()+'Geolife_Trajectories/'

output={}
output["bike"]  =np.array([[1,0,0,0,0,0,0]])
output["bus"]   =np.array([[0,1,0,0,0,0,0]])
output["car"]   =np.array([[0,0,1,0,0,0,0]])
output["subway"]=np.array([[0,0,0,1,0,0,0]])
output["taxi"]  =np.array([[0,0,0,0,1,0,0]])
output["train"] =np.array([[0,0,0,0,0,1,0]])
output["walk"]  =np.array([[0,0,0,0,0,0,1]])

class_X=np.zeros((1,15,2))
class_Y=np.zeros((1,7))


for myfile in tqdm(os.listdir(get_geolife_path()+'Geolife_datasets/')):
  if myfile.endswith('X.npy'):
    if myfile != 'classificationX.npy' and myfile != 'datasetX.npy':
      means = myfile.split("_")[1].split("X.")[0]
      X = np.load(get_geolife_path()+'Geolife_datasets/'+myfile)
      Y = np.repeat(output[means], X.shape[0], axis=0) 
      print(means)
      class_X = np.append(class_X, X, axis=0)
      class_Y = np.append(class_Y, Y, axis=0)

class_X = np.delete(class_X, 0, axis=0)
class_Y = np.delete(class_Y, 0, axis=0)
print(class_X.shape, class_Y.shape)
print(class_X[:2])
print(class_Y[:2])
np.save(get_geolife_path()+'Geolife_datasets/classificationX.npy', class_X)
np.save(get_geolife_path()+'Geolife_datasets/classificationY.npy', class_Y)