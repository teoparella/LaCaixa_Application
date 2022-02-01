import numpy as np
from pycbc.filter import matched_filter
from pycbc.psd import inverse_spectrum_truncation
from pycbc.psd import interpolate
import pycbc.frame
from pycbc.types import TimeSeries
from pycbc.types import zeros
from pycbc.types import Array
from pycbc.types import FrequencySeries
from pycbc.detector import Detector, effective_distance
from pycbc.waveform import get_td_waveform
import h5py
import sys

#Directions

OutputData = '/nfs/virgo/tparella/Analysis/GenerateSpectrograms/Output1/OutputData/'
OutputL1 = '/nfs/virgo/tparella/Analysis/GenerateSpectrograms/Output1/OutputL1/'
OutputH1 = '/nfs/virgo/tparella/Analysis/GenerateSpectrograms/Output1/OutputH1/'

#Parameters

MaxNumSignals=900001
Detector1='L1'
Detector2='H1'
Channel1='L1:DCS-CALIB_STRAIN_C01'
Channel2='H1:DCS-CALIB_STRAIN_C01'
Aproximants = ['IMRPhenomPv2','IMRPhenomPv3HM']   #'IMRPhenomNSBH'
Sfreq=16384
Lowerf=25
SignalsPerFile=1000
MaximumDist=300          # efficient eff. dist aprox 300 pc. Forcing 350 pc. Max detection 477 pc.
TGps=1186537603

#GW Variables

Partitions = 30
ChirpMass=[1,61]
EffectiveDistance=[1,1201]
DeltaChirp=(ChirpMass[1]-ChirpMass[0])/Partitions
DeltaDistance = (EffectiveDistance[1]-EffectiveDistance[0])/Partitions

Qfact=[0.1,1]
Distance=np.array([1,50])
RA=np.array([0,2*np.pi])
Dec=np.array([-np.pi/2,np.pi/2])
Pol=np.array([0,np.pi])
Inc=np.array([0,np.pi/2.])

#Time localization of signals
j=2 #First signals from j*TimeStep to j*TimeStep+TimeStep
TimeStep=5
Time_lenght=5000+16
Cut=2


###########################################################################################################################################################################

Index='For Condor purposes'

CurrentChirp=Index[0]
CurrentEffectiveDistance=Index[1]

if (MaxNumSignals-SignalsPerFile*(Index))<SignalsPerFile:
    SignalsPerFile=MaxNumSignals-SignalsPerFile*(Index)
   
if (MaxNumSignals-SignalsPerFile*(Index))<0:
    sys.exit()

ts_zeros1=np.zeros(int(Time_lenght*Sfreq))
ts_zeros1=TimeSeries(ts_zeros1,delta_t=1./Sfreq)

ts_zeros2=np.zeros(int(Time_lenght*Sfreq))
ts_zeros2=TimeSeries(ts_zeros2,delta_t=1./Sfreq)

Detec1=Detector(Detector1)
Detec2=Detector(Detector2)
Dat=np.zeros((SignalsPerFile,14))

for Idx in range(0,SignalsPerFile):
    Qfactor=np.random.uniform(Qfact[0],Qfact[1])
    RandomChirp=CurrentChirp+np.random.uniform(0,1)*DeltaChirp
    m1=RandomChirp*(((1+Qfactor)**(1/5))/(Qfactor**(3/5)))
    m2=Qfactor*m1
    d=np.random.uniform(EffectiveDistance[0],CurrentEffectiveDistance)
    ra=np.random.uniform(RA[0],RA[1])
    dec=np.random.uniform(Dec[0],Dec[1])
    pol=np.random.uniform(Pol[0],Pol[1])
    incl=np.random.uniform(Inc[0],Inc[1])

    fp1,fc1=Detec1.antenna_pattern(ra,dec,pol,TGps)
    
    EffecDist=effective_distance(d,incl,fp1,fc1)
    
    while EffecDist>CurrentEffectiveDistance or EffecDist<(CurrentEffectiveDistance-DeltaDistance):
        ra=np.random.uniform(RA[0],RA[1])
        dec=np.random.uniform(Dec[0],Dec[1])
        pol=np.random.uniform(Pol[0],Pol[1])
        incl=np.random.uniform(Inc[0],Inc[1])   
        fp1,fc1=Detec1.antenna_pattern(ra,dec,pol,TGps)
        d=np.random.uniform(EffectiveDistance[0],CurrentEffectiveDistance)
        EffecDist=effective_distance(d,incl,fp1,fc1)

    fp2,fc2=Detec2.antenna_pattern(ra,dec,pol,TGps)

    hpl, hcr = get_td_waveform(approximant=np.random.choice(Aproximants), mass1=m1, mass2=m2, distance=d,inclination=incl, delta_t=1./Sfreq, f_lower=Lowerf) 
    
    hp1=hpl*fp1+hcr*fc1
    hp2=hpl*fp2+hcr*fc2 

    Duration=float(hp1.end_time-hp1.start_time)

    print(hp1.start_time)
    print(hp1.end_time)
    try:
        hp1=hp1.time_slice(hp1.end_time-Cut,hp1.end_time)
    except:
        print('hp1 aborted at Idx : ', Idx)
        print('hp1 start time : ', hp1.start_time)
        print('hp1 end time : ', hp1.end_time)
        print('hp1 end - cut : ', hp1.end_time-Cut)
        sys.exit()
    try:
        hp2=hp2.time_slice(hp2.end_time-Cut,hp2.end_time)
    except:
        print('hp2 aborted at Idx : ', Idx)
        print('hp2 start time : ', hp2.start_time)
        print('hp2 end time : ', hp2.end_time)
        print('hp2 end - cut : ', hp2.end_time-Cut)
        sys.exit()
 
    if hp1.start_time < 0:
        hp1.start_time+=np.abs(hp1.start_time)
        hp2.start_time+=np.abs(hp2.start_time)
    else:
        hp1.start_time-=hp1.start_time
        hp2.start_time-=hp2.start_time
    
    LimiteSup=float(TimeStep-hp1.end_time)

    random=np.random.random()*LimiteSup

    Delay12=Detec1.time_delay_from_location(Detec2.location,ra,dec,TGps)

    hp1.start_time+=random+j*TimeStep
    hp2.start_time+=random+j*TimeStep+Delay12

    Dat[Idx,3]=hp1.start_time
    Dat[Idx,4]=hp1.end_time

    Dat[Idx,5]=hp2.start_time
    Dat[Idx,6]=hp2.end_time

    Num=(hp1.start_time)*Sfreq
    Num=int(np.floor(float(Num)))
    hp1.prepend_zeros(Num)
    hp1.append_zeros(Time_lenght*Sfreq-len(hp1))

    Num=(hp2.start_time)*Sfreq
    Num=int(np.floor(float(Num)))
    hp2.prepend_zeros(Num)
    hp2.append_zeros(Time_lenght*Sfreq-len(hp2))

    hp1=TimeSeries(hp1,delta_t=1./Sfreq,epoch=0)
    hp2=TimeSeries(hp2,delta_t=1./Sfreq,epoch=0)

    ts_zeros1=hp1+ts_zeros1
    ts_zeros2=hp2+ts_zeros2  

    del hp1,hp2

    Dat[Idx,0]=m1
    Dat[Idx,1]=m2
    Dat[Idx,2]=d
    Dat[Idx,7]=j*TimeStep
    Dat[Idx,8]=TimeStep
    Dat[Idx,9]=ra
    Dat[Idx,10]=dec
    Dat[Idx,11]=pol
    Dat[Idx,12]=incl
    Dat[Idx,13]=Duration
    Dat[Idx,14]=RandomChirp
    Dat[Idx,15]=EffecDist
    j+=1



g1 = h5py.File('%sInformation_%s_%s.h5' % (OutputData,Index[0],Index[1]), 'w')
g1.create_dataset('Mass1',data=Dat[:,0])
g1.create_dataset('Mass2',data=Dat[:,1])
g1.create_dataset('Luminosity_distance',data=Dat[:,2])
g1.create_dataset('hp1_start_time',data=Dat[:,3])
g1.create_dataset('hp1_end_time',data=Dat[:,4])
g1.create_dataset('hp2_start_time',data=Dat[:,5])
g1.create_dataset('hp2_end_time',data=Dat[:,6])
g1.create_dataset('Beginning_of_interval',data=Dat[:,7])
g1.create_dataset('Interval_lenght',data=Dat[:,8])
g1.create_dataset('RA',data=Dat[:,9])
g1.create_dataset('Declination',data=Dat[:,10])
g1.create_dataset('Polarization',data=Dat[:,11])
g1.create_dataset('Inclination', data=Dat[:,12])
g1.create_dataset('Duration',data=Dat[:,13])
g1.create_dataset('Chirp_Mass',data=Dat[:,14])
g1.create_dataset('Effective_Distance',data=Dat[:,15])
g1.create_dataset('NTemplates',data=SignalsPerFile)

pycbc.frame.write_frame(OutputL1+str(Index[0])+"_"+str(Index[1])+".gwf",Channel1, ts_zeros1)
pycbc.frame.write_frame(OutputH1+str(Index[0])+"_"+str(Index[1])+".gwf",Channel2, ts_zeros2)





