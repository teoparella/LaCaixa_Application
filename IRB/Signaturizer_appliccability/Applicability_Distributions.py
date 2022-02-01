
import h5py 
import os
import numpy as np
import random
from signaturizer import Signaturizer
import matplotlib.pyplot as plt

sign = Signaturizer('GLOBAL')


CC_path='/aloy/home/tparella/Bioactivity/databases/libraries/ChemicalChecker/CC_data.h5'
Zinc_path = '/aloy/home/tparella/Bioactivity/databases/libraries/Zinc/Not_DrugLike/dataframes/'
Chembl_path='/aloy/home/tparella/Bioactivity/databases/libraries/Chembl/Chembl_data.h5'

Zinc_df_Names=Zinc_dirs = [_ for _ in os.listdir(Zinc_path) if _.endswith('.h5')]
Zinc_df_Names.sort()
print(Zinc_df_Names)
Under200Dalton=Zinc_df_Names[:11]
print('Under200Dalton: ',Under200Dalton)
Over500Dalton=Zinc_df_Names[19:]
print('Over500Dalton: ',Over500Dalton)
HighReactivity=Zinc_df_Names[10:19]
HighReactivity.append(Zinc_df_Names[-1])


Each_number=10000

f1=h5py.File(CC_path,'r')
CC_smiles=f1['smiles'][:]
CC_smiles=random.sample(list(CC_smiles),Each_number)
f1.close()

f2=h5py.File(Chembl_path,'r')
Chembl_smiles=f2['smiles'][:1000000]
#Chembl_smiles=random.sample(list(Chembl_smiles),Each_number)
f2.close()

Under200Dalton_morgans=np.array([])
for i in Under200Dalton:
    f1=h5py.File(Zinc_path+str(i),'r')
    a_morgans=np.array(f1['smiles'][:Each_number])
    try:
        a_morgans=random.sample(a_morgans,Each_number)
    except:
        pass
    a_morgans=np.squeeze(a_morgans)
    if i=='AA.h5':
        Under200Dalton_morgans=a_morgans
    else:
        Under200Dalton_morgans=np.concatenate((Under200Dalton_morgans,a_morgans))
    f1.close()
print('Under200Dalton_morgans.shape = ', Under200Dalton_morgans.shape)



Over500Dalton_morgans=np.array([])
for i in Over500Dalton:
    f1=h5py.File(Zinc_path+str(i),'r')
    a_morgans=np.array(f1['smiles'][:Each_number])
    try:
        a_morgans=random.sample(a_morgans,Each_number)
    except:
        pass
    a_morgans=np.squeeze(a_morgans)
    if i=='KA.h5':
        Over500Dalton_morgans=a_morgans
    else:
        Over500Dalton_morgans=np.concatenate((Over500Dalton_morgans,a_morgans))
    f1.close()
print('Over500Dalton_morgans.shape = ', Over500Dalton_morgans.shape)


HighReactivity_morgans=np.array([])
for i in HighReactivity:
    f1=h5py.File(Zinc_path+str(i),'r')
    try:
        a_morgans=np.array(f1['smiles'][:Each_number])
    except:
        continue
    try:
        a_morgans=random.sample(a_morgans,Each_number)
    except:
        pass
    a_morgans=np.squeeze(a_morgans)
    if i=='AK.h5':
        HighReactivity_morgans=a_morgans
    else:
        HighReactivity_morgans=np.concatenate((HighReactivity_morgans,a_morgans))
    f1.close()
print('HighReactivity_morgans.shape = ', HighReactivity_morgans.shape)

Over500Dalton_morgans =random.sample(list(Over500Dalton_morgans),Each_number)
Under200Dalton_morgans =random.sample(list(Under200Dalton_morgans),Each_number)
HighReactivity_morgans =random.sample(list(HighReactivity_morgans),Each_number)


app_CC = np.array(sign.predict(CC_smiles).applicability)
app_Chembl = np.array(sign.predict(Chembl_smiles).applicability)
app_Under_200D =np.array(sign.predict(Under200Dalton_morgans).applicability)
app_Over_500D =np.array(sign.predict(Over500Dalton_morgans).applicability)
app_HighReactivity =np.array(sign.predict(HighReactivity_morgans).applicability)
'''
plt.imshow(app_CC)
plt.show()
plt.imshow(app_Chembl)
plt.show()
plt.imshow(app_Under_200D)
plt.show()
plt.imshow(app_Over_500D)
plt.show()
plt.imshow(app_HighReactivity)
plt.show()'''


save_path='/aloy/home/tparella/Bioactivity/Applicability/Data/'
np.save(save_path+'app_CC.npy', app_CC)
print('1')
np.save(save_path+'app_Chembl.npy', app_Chembl)
print('2')
np.save(save_path+'app_Under_200D.npy', app_Under_200D)
print('2')
np.save(save_path+'app_Over_500D.npy', app_Over_500D)
print('2')
np.save(save_path+'app_HighReactivity.npy', app_HighReactivity)


import seaborn as sns
for itera in range(25):
    print(itera)
    fig, axs = plt.subplots(1,2, figsize=(30,10))
    col=app_CC[:,itera]
    ax1 = sns.distplot(col,bins=150, ax=axs[0], label='CC', color='r')
    col2=app_Chembl[:,itera]
    ax1 = sns.distplot(col2,bins=150, ax=axs[0], label='CC', color='g')
    col3=app_Under_200D[:,itera]
    ax1 = sns.distplot(col3,bins=150, ax=axs[0], label='CC', color='blue')
    col4=app_Over_500D[:,itera]
    #ax1 = sns.distplot(col4,bins=150, ax=axs[0], label='CC', color='b')
    col5=app_HighReactivity[:,itera]
    #ax1 = sns.distplot(col5,bins=150, ax=axs[0], label='CC', color='b')
    plt.show()

