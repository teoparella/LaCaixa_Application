#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc


# # Morgan ROC

# In[2]:


import PipelineProfiler

modelpath = '/aloy/home/tparella/Bioactivity/Models/CC_to_ZincBioinactive/Analysis/models/dummy_model/lda_gradboost/lda_gradboost_40k_10k_1d_12p_2cv/dummy_0.sav'
loaded_model = pickle.load(open(modelpath, 'rb'))

profiler_data = PipelineProfiler.import_autosklearn(loaded_model)
PipelineProfiler.plot_pipeline_matrix(profiler_data)


# In[3]:


CC_path = '/aloy/home/tparella/Bioactivity/Models/CC_to_ZincBioinactive_A1/Data/Actives_CC/CC_data.h5'
Zinc_path='/aloy/home/tparella/Bioactivity/Models/CC_to_ZincBioinactive_A1/Data/Inactives_Zinc/dataframes/'
Zinc_dirs = [_ for _ in os.listdir(Zinc_path) if _.endswith('.h5')]

#CC_path='/aloy/home/tparella/Bioactivity/Models/CC_to_ZincBioinactive_B4/Data/Chembl/Chembl_data.h5'

f1=h5py.File(CC_path,'r')
CC_morgans1=np.array(f1['morgans'][:500000])
CC_morgans1=np.squeeze(np.vsplit(CC_morgans1,CC_morgans1.shape[0]))
CC_morgans=random.sample(list(CC_morgans1),10000)

CC_A1_aux=np.array(f1['A1_sign_3'][:500000])
CC_A1_aux=np.squeeze(np.vsplit(CC_A1_aux,CC_A1_aux.shape[0]))
CC_A1=random.sample(list(CC_A1_aux),10000)
print('CC loaded')

Zinc_samples=10000
Zinc_morg=[]
Zinc_A1=[]
samples_per_tranch=5000
#A1_sign_3
for i in Zinc_dirs:
    f2=h5py.File(Zinc_path+i,'r')
    try:
        Zinc_morg_help=np.array(f2['morgans'][:samples_per_tranch])
        Zinc_A1_help=np.array(f2['A1_sign_3'][:samples_per_tranch])
        Zinc_morg_help=np.vsplit(Zinc_morg_help,Zinc_morg_help.shape[0])
        Zinc_A1_help=np.vsplit(Zinc_A1_help,Zinc_A1_help.shape[0])
        Zinc_morg+=random.sample(Zinc_morg_help,len(Zinc_morg_help))
        Zinc_A1+=random.sample(Zinc_A1_help,len(Zinc_A1_help))
        del(Zinc_A1_help)
    except:
        print('{} less '.format(i)+'than {} compounds'.format(samples_per_tranch))
        try:
            Zinc_A1_help=np.array(f2['morgan'][:])
            Zinc_A1_help=np.array(f2['A1_sign_3'][:])
            Zinc_A1_help=np.vsplit(Zinc_A1_help,Zinc_A1_help.shape[0])
            Zinc_A1+=random.sample(Zinc_A1_help,len(Zinc_A1_help))
            del(Zinc_A1_help)
        except:
            print('file {} not loaded'.format(i))
    f2.close()
#print(Zinc_A1)
Zinc_A1=random.sample(Zinc_A1,Zinc_samples)
Zinc_morg=random.sample(Zinc_morg,Zinc_samples)
'''
f2=h5py.File(Zinc_path,'r')
Zinc_morgans1=np.array(f2['morgans'][:])
Zinc_morgans1=np.squeeze(np.vsplit(Zinc_morgans1,Zinc_morgans1.shape[0]))
Zinc_morgans=random.sample(list(Zinc_morgans1),10000)
'''
print('Zinc loaded')

print(len(CC_morgans))
print(CC_morgans[:3])
print(len(CC_A1))
print(CC_A1[:3])
print(len(Zinc_A1))
print(Zinc_A1[:3])
print(len(Zinc_morg))
print(Zinc_morg[3])


# In[4]:


####        CC = 1.0    ,    Zinc = 0.0        ####

Predicted_CC = loaded_model.predict_proba(CC_morgans)[:,1]
Predicted_Zinc = loaded_model.predict_proba(np.squeeze(np.array(Zinc_morg)))[:,1]
print(Predicted_CC)
print(Predicted_Zinc)
True_CC = np.ones(len(Predicted_CC))


# In[5]:


print(len(Predicted_Zinc))


# In[13]:


Predicted_data=list(Predicted_CC)+list(Predicted_Zinc)
CC_ones=list(np.ones((len(Predicted_CC))))
Zinc_Zeros=list(np.zeros(len(Predicted_Zinc)))
True_data=CC_ones+Zinc_Zeros
fpr, tpr, _ = roc_curve(True_data,Predicted_data)
roc_auc = auc(fpr, tpr)


# In[7]:


CC_path = '/aloy/home/tparella/Bioactivity/Models/CC_to_ZincBioinactive_B4/Data/Actives_CC/CC_data_B4.h5'
Zinc_path='/aloy/home/tparella/Bioactivity/Models/CC_to_ZincBioinactive_B4/Data/Inactives_Zinc/dataframes/'
Zinc_dirs = [_ for _ in os.listdir(Zinc_path) if _.endswith('.h5')]

#CC_path='/aloy/home/tparella/Bioactivity/Models/CC_to_ZincBioinactive_B4/Data/Chembl/Chembl_data.h5'

f1=h5py.File(CC_path,'r')

CC_B4_aux=np.array(f1['B4_sign_3'][:])
CC_B4_aux=np.squeeze(np.vsplit(CC_B4_aux,CC_B4_aux.shape[0]))
CC_B4=random.sample(list(CC_B4_aux),10000)
print('CC loaded')

Zinc_samples=10000
Zinc_B4=[]
samples_per_tranch=5000
#A1_sign_3
for i in Zinc_dirs:
    f2=h5py.File(Zinc_path+i,'r')
    try:
        Zinc_B4_help=np.array(f2['B4_sign_3'][:samples_per_tranch])
        Zinc_B4_help=np.vsplit(Zinc_B4_help,Zinc_B4_help.shape[0])
        Zinc_B4+=random.sample(Zinc_B4_help,len(Zinc_B4_help))
        del(Zinc_B4_help)
    except:
        print('{} less '.format(i)+'than {} compounds'.format(samples_per_tranch))
        try:
            Zinc_B4_help=np.array(f2['B4_sign_3'][:])
            Zinc_B4_help=np.vsplit(Zinc_B4_help,Zinc_B4_help.shape[0])
            Zinc_B4=random.sample(Zinc_B4_help,len(Zinc__help))
            del(Zinc_B4_help)
        except:
            print('file {} not loaded'.format(i))
    f2.close()
#print(Zinc_A1)
Zinc_B4=random.sample(Zinc_B4,Zinc_samples)
'''
f2=h5py.File(Zinc_path,'r')
Zinc_morgans1=np.array(f2['morgans'][:])
Zinc_morgans1=np.squeeze(np.vsplit(Zinc_morgans1,Zinc_morgans1.shape[0]))
Zinc_morgans=random.sample(list(Zinc_morgans1),10000)
'''
print('Zinc loaded')

print(len(CC_B4))
print(CC_B4[:3])
print(len(Zinc_B4))
print(Zinc_B4[:3])


# In[8]:


import PipelineProfiler

modelpathB4 = '/aloy/home/tparella/Bioactivity/Models/CC_to_ZincBioinactive_B4/Analysis/AutoSKlearn_0/B4_40k_10k_12h_autosklearn.sav'
loaded_modelB4 = pickle.load(open(modelpathB4, 'rb'))

profiler_dataB4 = PipelineProfiler.import_autosklearn(loaded_modelB4)
PipelineProfiler.plot_pipeline_matrix(profiler_dataB4)


# In[9]:


Predicted_CCB4 = loaded_modelB4.predict_proba(CC_B4)[:,1]
Predicted_ZincB4 = loaded_modelB4.predict_proba(np.squeeze(np.array(Zinc_B4)))[:,1]
print(Predicted_CCB4)
print(Predicted_ZincB4)
True_CC = np.ones(len(Predicted_CCB4))


# In[10]:


Predicted_dataB4=list(Predicted_CCB4)+list(Predicted_ZincB4)
CC_ones=list(np.ones((len(Predicted_CCB4))))
Zinc_Zeros=list(np.zeros(len(Predicted_ZincB4)))
True_data=CC_ones+Zinc_Zeros
fpr3, tpr3, _ = roc_curve(True_data,Predicted_dataB4)
roc_auc3 = auc(fpr3, tpr3)


# # A1 ROC 

# In[11]:


import PipelineProfiler

modelpath2 = '/aloy/home/tparella/Bioactivity/Models/CC_to_ZincBioinactive_A1/Analysis/AutoSKlearn_0/First_A1_40k_10k_12h_autosklearn.sav'
loaded_model_2 = pickle.load(open(modelpath2, 'rb'))

profiler_data2 = PipelineProfiler.import_autosklearn(loaded_model_2)
PipelineProfiler.plot_pipeline_matrix(profiler_data2)


# In[12]:



Predicted_CC2 = loaded_model_2.predict_proba(CC_A1)[:,1]
Predicted_Zinc2 = loaded_model_2.predict_proba(np.squeeze(np.array(Zinc_A1)))[:,1]
Predicted_data2=list(Predicted_CC2)+list(Predicted_Zinc2)
CC_ones=list(np.ones((len(Predicted_CC2))))
Zinc_Zeros=list(np.zeros(len(Predicted_Zinc2)))
True_data=CC_ones+Zinc_Zeros
fpr2, tpr2, _ = roc_curve(True_data,Predicted_data2)
roc_auc2 = auc(fpr2, tpr2)


# # Plot ROC

# In[16]:


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='red',lw=lw, label='Morgan fingerprints ROC (AUC = %0.2f)' % roc_auc) #darkorange
plt.plot(fpr2, tpr2, color='blue',lw=lw, label='A1 Signature 3 ROC (area = %0.2f)' % roc_auc2) #darkorange
plt.plot(fpr3, tpr3, color='green',lw=lw, label='B4 Signature 3 ROC (area = %0.2f)' % roc_auc3) #darkorange
plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--') #navy
plt.xlim([0, 1.])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[5]:


zero_counter=0
one_counter=0
other_counter=0

for i in Predicted_CC:
    if i== 0.0:
        zero_counter +=1
    elif i == 1.0:
        one_counter+=1
    else:
        other_counter+=1
        
print('Total Samples = ' + str(len(Predicted_CC)))
print('Zinc Samples = ' + str(zero_counter))
print('Zinc fraction = ' + str(zero_counter/len(Predicted_CC)))
print('CC Samples = ' + str(one_counter))
print('CC fraction = ' + str(one_counter/len(Predicted_CC)))
print('Other Samples = ' + str(other_counter))
print('Other fraction = ' + str(other_counter/len(Predicted_CC)))


# In[ ]:


#0-200

Total Samples = 200000
Zinc Samples = 196364
Zinc fraction = 0.98182
CC Samples = 3636
CC fraction = 0.01818
Other Samples = 0
Other fraction = 0.0


#200:

Total Samples = 151045
Zinc Samples = 149110
Zinc fraction = 0.9871892482372803
CC Samples = 1935
CC fraction = 0.01281075176271972
Other Samples = 0
Other fraction = 0.0


# In[3]:


Zinc_samples=196364+149110
CC_samples=3636 + 1935

Zinc_frac= Zinc_samples / len(CC_morgans1)

CC_frac= CC_samples / len(CC_morgans1)

print(Zinc_samples)
print(CC_samples)
print(Zinc_frac)
print(CC_frac)


# In[6]:


################# 0-200k 
Total Samples = 200000
Zinc Samples = 27170
Zinc fraction = 0.13585
CC Samples = 172830
CC fraction = 0.86415
Other Samples = 0
Other fraction = 0.0
################# 200-400k 
Total Samples = 200000
Zinc Samples = 27264
Zinc fraction = 0.13632
CC Samples = 172736
CC fraction = 0.86368
Other Samples = 0
Other fraction = 0.0

################# 400-600k 
Total Samples = 200000
Zinc Samples = 27488
Zinc fraction = 0.13744
CC Samples = 172512
CC fraction = 0.86256
Other Samples = 0
Other fraction = 0.0

################# 600-800k 

Total Samples = 200000
Zinc Samples = 26923
Zinc fraction = 0.134615
CC Samples = 173077
CC fraction = 0.865385
Other Samples = 0
Other fraction = 0.0

################# 800k 
:Total Samples = 135476
Zinc Samples = 18239
Zinc fraction = 0.1346290117806844
CC Samples = 117237
CC fraction = 0.8653709882193156
Other Samples = 0
Other fraction = 0.0




# In[5]:


Zinc_samples=27170+27264+27488+26923+18239
CC_samples=172830+172736+172512+173077+117237

Zinc_frac= Zinc_samples / len(CC_morgans1)

CC_frac= CC_samples / len(CC_morgans1)

print(Zinc_samples)
print(CC_samples)
print(Zinc_frac)
print(CC_frac)


# # Precision recall curve

# In[18]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score


# In[19]:


average_precision = average_precision_score(True_data,Predicted_data)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

average_precision = average_precision_score(True_data,Predicted_data2)

print('Average precision-recall score 2: {0:0.2f}'.format(
      average_precision))

average_precision = average_precision_score(True_data,Predicted_dataB4)

print('Average precision-recall score 2: {0:0.2f}'.format(
      average_precision))


# In[24]:


precision, recall, thresholds = precision_recall_curve(True_data, Predicted_data)
precision2, recall2, thresholds2 = precision_recall_curve(True_data, Predicted_data2)
precision3, recall3, thresholds3 = precision_recall_curve(True_data, Predicted_dataB4)


# In[27]:


plt.figure()
lw = 2
plt.plot(recall, precision, color='red',lw=lw, label='Morgan fingerprints (PR score = 0.94)') #darkorange
plt.plot(recall2, precision2, color='blue',lw=lw, label='A1 Signature 3 (PR score = 0.88)' % roc_auc2) #darkorange
plt.plot(recall3, precision3, color='green',lw=lw, label='B4 Signature 3 (PR score = 0.94)' % roc_auc2) #darkorange
plt.plot([0, 1], [0.5, 0.5], color='grey', lw=lw, linestyle='--') #navy
plt.xlim([0, 1.])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
#plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[29]:


plt.figure()
lw = 2
plt.plot(recall, precision, color='red',lw=lw, label='CC') #darkorange
plt.plot(recall2, precision2, color='blue',lw=lw, label='Zinc \"Not druglike\"' ) #darkorange
plt.plot(recall3, precision3, color='green',lw=lw, label='Chembl' ) #darkorange
plt.plot([0, 1], [0.5, 0.5], color='grey', lw=lw, linestyle='--') #navy
plt.xlim([0, 1.])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
#plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

