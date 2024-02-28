from distillation_experiments import run_experiments
from classification import transportation_classifier
from plot_course import plot_random_course

'''
Runs experiments in regards to training student models by various methods 
(self-training, knowledge-distillation, double-distillation, fine-tuning)
and creates the tensorflow and tensorflow-lite models.
Requires the appropriate input .npy files in data_path/Geolife/Geolife_datasets (see readme.md)

Also expects a 'Teacher.h5' model file in data_path/Geolife/Geolife_models.
Set the data_path in the 'geolife_data_utilities.py' file.
'''
run_experiments(filename='experiment_outputs', teacher_model_name='Teacher') 


'''
Trains the transportation means classifier.
Requires the appropriate input .npy files in data_path/Geolife/Geolife_datasets (see readme.md)
'''
transportation_classifier(undersampling=True, train_new=True)


'''
Takes a random input sample, predicts the transportation means and by using the appropriate model
predicts the next movement. Then it plots the input movement as well as the true and predicted
movement on a cartesian field.

Requires the appropriate input .npy files in data_path/Geolife/Geolife_datasets (see readme.md)
'''
plot_random_course()