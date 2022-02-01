
import pandas as pd
import autosklearn.classification  #or regression
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import os
import h5py
import random
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score


## Load Data

CC_path = '/aloy/home/tparella/Bioactivity/Models/CC_to_ZincBioinactive/Data/Actives_CC/CC_data.h5'
Zinc_path = '/aloy/home/tparella/Bioactivity/Models/CC_to_ZincBioinactive/Data/Inactives_Zinc/Zinc_morgans_2.h5'


f1=h5py.File(CC_path,'r')
CC_morgans1=np.array(f1['morgans'])
CC_morgans1=np.vsplit(CC_morgans1,CC_morgans1.shape[0])
CC_morgans=random.sample(CC_morgans1,50000)
del(CC_morgans1)
f1.close()

print('CC_Data '+ str(len(CC_morgans))+' compounds.')

f2=h5py.File(Zinc_path,'r')
Zinc_morgans1=np.array(f2['morgans'])
Zinc_morgans1=np.vsplit(Zinc_morgans1,Zinc_morgans1.shape[0])
Zinc_morgans=random.sample(Zinc_morgans1,15000)
del(Zinc_morgans1)
f2.close()

print('ZincData '+ str(len(Zinc_morgans))+' compounds.')



Zinc_To_CC_Proportion= 1

if len(Zinc_morgans)*Zinc_To_CC_Proportion >= len(CC_morgans):
    Zinc_morgans=random.sample(Zinc_morgans, int(len(CC_morgans)*Zinc_To_CC_Proportion))
    print('CC compounds:  '+str(len(CC_morgans)))
    print('Zinc compounds restricted to:  '+str(len(Zinc_morgans)))

elif len(Zinc_morgans)*Zinc_To_CC_Proportion < len(CC_morgans):
    CC_morgans=random.sample(CC_morgans,int(len(Zinc_morgans) / Zinc_To_CC_Proportion))
    print('CC compounds restricted to:  '+str(len(CC_morgans)))
    print('Zinc compounds:  '+str(len(Zinc_morgans)))


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    print('shuffle permutation :', p)
    a_ret=[]
    b_ret=[]
    for i in p:
        a_ret.append(a[i])
        b_ret.append(b[i])
    return a_ret, b_ret

Zinc_Zeros=np.zeros(len(Zinc_morgans))
CC_Ones=np.ones(len(CC_morgans))

Molecules=np.squeeze(np.concatenate((Zinc_morgans,CC_morgans)))
Indexes=np.concatenate((Zinc_Zeros,CC_Ones))

print(np.squeeze(Molecules).shape)
print(Indexes.shape)

Vector_samples,Vector_indexes = unison_shuffled_copies(Molecules,Indexes)


X_train=Vector_samples
Y_train=Vector_indexes


print(len(Train_samples[0]))



Test_split= 0.2

Train_samples=Vector_samples[int(len(Vector_samples)*Test_split):]
Train_indexes=Vector_indexes[int(len(Vector_indexes)*Test_split):]
Test_samples=Vector_samples[:int(len(Vector_samples)*Test_split)]
Test_indexes=Vector_indexes[:int(len(Vector_indexes)*Test_split)]

print('Train samples =  ',len(Train_samples))
print('Train indexes =  ',len(Train_indexes))
print('Test samples =  ',len(Test_samples))
print('Test indexes =  ',len(Test_indexes))


# # CROSSVALIDATED SKLEARN


current=10

    
print('split {}'.format(current))
    
model = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task = int(60*5), n_jobs=-1, include_preprocessors = ["no_preprocessing"], ensemble_size=1, initial_configurations_via_metalearning=0,resampling_strategy='cv',resampling_strategy_arguments={'folds': 10},include_estimators=['gradient_boosting','lda'])
model.fit(X_train,Y_train)
    

    
filename = '/aloy/home/tparella/Bioactivity/Models/CC_to_ZincBioinactive/Analysis/models/dummy_model/'
pickle.dump(model, open(filename+'dummy_'+str(current)+'.sav', 'wb'))


# # Autosklearn without crossvalidation


model = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task = int(5*60), n_jobs=3, include_preprocessors = ["no_preprocessing"], ensemble_size=1, initial_configurations_via_metalearning=0)  #include_preprocessors = []    #Or Regressor


print('training model')
model.fit(Train_samples,Train_indexes)
print('model trained')

print(model.sprint_statistics())

print(model.sprint_statistics())

print(model.show_models())


from sklearn.metrics import accuracy_score

pred= model.predict(Test_samples)

acc = accuracy_score(Test_indexes,list(pred))

print('accuracy = ', acc)


model.show_models()

model.get_models_with_weights()

model.leaderbord()

filename = '/aloy/home/tparella/Bioactivity/Models/CC_to_ZincBioinactive/Analysis/models/dummy_model.sav'
pickle.dump(model, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
print(loaded_model)

#loaded_model.leaderboard()


# # VISUALIZE MODELS 


import PipelineProfiler

modelpath = '/aloy/home/tparella/Bioactivity/Models/CC_to_ZincBioinactive/Analysis/models/dummy_model/dummy_0.sav'
loaded_model = pickle.load(open(modelpath, 'rb'))


profiler_data = PipelineProfiler.import_autosklearn(loaded_model)
PipelineProfiler.plot_pipeline_matrix(profiler_data)






