import keras
import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger     #import earlystopping
from keras import optimizers
from keras import applications

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

from Generator_Sep import DataGenerator

Path='/nfs/virgo/tparella/SandBox/SandBox1/GW/Shuffle/FinalData/'

InputSh=[100, 400, 2] #Data Shape

Validation_split=0.1
ValShuf=42 #Seed for the Train/Validation split
Mpc=True #Multiprocessing
Max_Queue=15
Workers=5 #Maximum number of processes to spin up when using process-based threading
Epochs=30
PathSave='/nfs/virgo/tparella/GW_AI/CNN/CNN_L1H1/Weights/Weights-{epoch:02d}-V1H1.h5'
PathLogs='/nfs/virgo/tparella/GW_AI/CNN/CNN_L1H1/Logs/NetworkL1H1.log'
PathInfoSave='/nfs/virgo/tparella/GW_AI/CNN/CNN_L1H1/InfoSafe/InfoSafeL1H1.h5'
Lr=0.001
Optimizer=optimizers.Adam(learning_rate=Lr)

Start,End=0,3000

#########################
Files=np.arange(Start,End)

IndexTrain, IndexVal = train_test_split(Files,test_size=Validation_split, random_state=ValShuf)

Shape=(InputSh[0],InputSh[1],InputSh[2]) #We only have 1 channel for the data

Base_Model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= Shape)
x = Base_Model.output
x = GlobalAveragePooling2D()(x)
Output = Dense(1, activation= 'sigmoid')(x)
Model_Comp = Model(inputs = Base_Model.input, outputs = Output)

Model_Comp.compile(optimizer=Optimizer, loss='binary_crossentropy',metrics=['accuracy'])

Generator=DataGenerator(Path,IndexTrain)
GeneratorVal=DataGenerator(Path,IndexVal)
Checkpoint = ModelCheckpoint(PathSave, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
CSV= CSVLogger(PathLogs, separator=' ', append=True)

CallBacks=[Checkpoint,CSV]     #earlystopping?¿

Save=h5py.File(PathInfoSave,'w')
Save.create_dataset('Val_Index',data=IndexVal)
Save.create_dataset('Index',data=IndexTrain)
Save.attrs['Path']=str(Path)
Save.attrs['Total_Epochs']=Epochs
Save.attrs['TrainValSplit_Seed']=ValShuf
Save.attrs['lr']=Lr
Save.close()

Model_Comp.fit_generator(generator=Generator,validation_data=GeneratorVal,validation_steps=None,use_multiprocessing=Mpc,max_queue_size=Max_Queue,workers=Workers,verbose=1,epochs=Epochs,callbacks=CallBacks)

