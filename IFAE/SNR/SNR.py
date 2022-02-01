from pycbc.waveform import get_td_waveform #Get waveform in time domain
import numpy as np
import pycbc.frame #It's used to read the format of the files: gwf 
import h5py
from pycbc.types import TimeSeries, FrequencySeries, zeros, Array
from pycbc.detector import Detector
from pycbc.filter import resample_to_delta_t, sigma
from pycbc.psd import inverse_spectrum_truncation, interpolate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import tukey
from Functions import SNR_Est

###New: Estimated the optimal SNR using a home made function and applied a tukey window

###Paths of the files and parameters

OutputH5='/nfs/virgo/menendez/CNN/2Chann/ResNet50/InjectionTest/LowM/'
BkgPath1='/nfs/virgo/H1_O2/' #Path to the Background files
BkgPath2='/nfs/virgo/L1_O2/' #Path to the Background files

OutputShape=[32,100,400,2]
NFiles=50

FileName_1=np.array(['H1','H-H1_GWOSC_O2_16KHZ_R1-1186959360-4096.gwf'])
FileName_2=np.array(['L1','L-L1_GWOSC_O2_16KHZ_R1-1186959360-4096.gwf'])

Channel_1='H1:GWOSC-16KHZ_R1_STRAIN'
Channel_2='L1:GWOSC-16KHZ_R1_STRAIN'

Approx='IMRPhenomPv2' #Approximant
Sfreq=16384 #Sampling frequency
Lowerf=80 #Low frequency cutoff
FreqRange=[20,1024]
TGps=1186537603
TimeTemplate=50
TimeStep=5 #Lenght of the images in seconds
max_filter_duration=4
Time_lenght=120
Cut=4
Alpha=1./9. #Parameter for the shape of the Tukey window, alpha=0 step function, alpha=1, hann window


#Limits parameters
M1=np.array([0.19,2.0])
M2=np.array([0.19,2.0])
Distance=np.array([1,50])
RA=np.array([0,2*np.pi])
Dec=np.array([0,np.pi])
Pol=np.array([0,np.pi])
Inc=np.array([0,np.pi/2.])

SNR1=np.arange(1,50.5,0.5)
SNR2=np.arange(1,50.5,0.5)
Same=True

##########
Index='For Condor purposes'

St=float(FileName_1[1].split('.')[0].split('-')[2]) #Start time of the background file

Back1=pycbc.frame.read_frame(BkgPath1+FileName_1[1],Channel_1,St,St+Time_lenght) #Reading the Background file
Back2=pycbc.frame.read_frame(BkgPath2+FileName_2[1],Channel_2,St,St+Time_lenght) #Reading the Background file

PSD=Back1.psd(max_filter_duration)
psd = interpolate(PSD, Back1.delta_f)
max_filter_len = int(max_filter_duration * Back1.sample_rate)
psd1 = inverse_spectrum_truncation(psd,
                   max_filter_len=max_filter_len,
                   low_frequency_cutoff=FreqRange[0],
                   trunc_method='hann')

PSD=Back2.psd(max_filter_duration)
psd = interpolate(PSD, Back2.delta_f)
psd2 = inverse_spectrum_truncation(psd,
                   max_filter_len=max_filter_len,
                   low_frequency_cutoff=FreqRange[0],
                   trunc_method='hann')


Detec1=Detector(FileName_1[0])
Detec2=Detector(FileName_2[0])

NameIndx=np.arange(Index*NFiles,(Index+1)*NFiles)

for j in range(NFiles):
    Hf1=h5py.File(OutputH5+str(int(NameIndx[j]))+'.h5','w')

    Hf1.attrs['File_1']=FileName_1[1]
    Hf1.attrs['File_2']=FileName_2[1]
    GroupSav=np.empty((OutputShape[0],OutputShape[1],OutputShape[2],OutputShape[3]))
    SNR=np.empty((OutputShape[0],2))

    for i in range(OutputShape[0]):

        m1=np.random.uniform(M1[0],M1[1])
        m2=np.random.uniform(M2[0],M2[1])
        d=np.random.uniform(Distance[0],Distance[1])
        ra=np.random.uniform(RA[0],RA[1])
        dec=np.random.uniform(Dec[0],Dec[1])
        pol=np.random.uniform(Pol[0],Pol[1])
        incl=np.random.uniform(Inc[0],Inc[1])

        fp1,fc1=Detec1.antenna_pattern(ra,dec,pol,TGps)
        fp2,fc2=Detec2.antenna_pattern(ra,dec,pol,TGps)

        hpl, hcr = get_td_waveform(approximant=Approx, mass1=m1, mass2=m2, distance=d,inclination=incl, delta_t=1./Sfreq, f_lower=Lowerf) 
        
        hp1=hpl*fp1+hcr*fc1     
        hp2=hpl*fp2+hcr*fc2 

        Duration=float(hp1.end_time-hp1.start_time)    

        if Duration>TimeStep:
            hp1=hp1.time_slice(hp1.end_time-Cut,hp1.end_time)
            hp2=hp2.time_slice(hp2.end_time-Cut,hp2.end_time)
        
        Tuk=tukey(len(hp1),Alpha)
        hp1=hp1*Tuk
        hp2=hp2*Tuk

        if hp1.start_time < 0:
            hp1.start_time+=np.abs(hp1.start_time)
            hp2.start_time+=np.abs(hp2.start_time)
        else:
            hp1.start_time-=hp1.start_time
            hp2.start_time-=hp2.start_time

        LimiteSup=float(TimeStep-hp1.end_time)

        random=np.random.random()*LimiteSup

        Delay12=Detec1.time_delay_from_location(Detec2.location,ra,dec,TGps)

        hp1.start_time+=random+TimeTemplate
        hp2.start_time+=random+TimeTemplate+Delay12

        Num=(hp1.start_time)*Sfreq
        Num=int(np.floor(float(Num)))
        hp1.prepend_zeros(Num)
        hp1.append_zeros(Time_lenght*Sfreq-len(hp1))

        Num=(hp2.start_time)*Sfreq
        Num=int(np.floor(float(Num)))
        hp2.prepend_zeros(Num)
        hp2.append_zeros(Time_lenght*Sfreq-len(hp2))

        snr_opt_1 = SNR_Est(hp1, psd1, Lowerf,FreqRange[1]) 
        snr_opt_2 = SNR_Est(hp2, psd2, Lowerf,FreqRange[1]) 

        if Same:
            snr1=np.random.choice(SNR1)
            snr2=snr1
        else:
            snr1=np.random.choice(SNR1)
            snr2=np.random.choice(SNR2)

        SNR[i,0]=snr1
        SNR[i,1]=snr2

        Ratio1=snr1/snr_opt_1
        Ratio2=snr2/snr_opt_2

        hp1=Ratio1*hp1
        hp2=Ratio2*hp2

        hp1=TimeSeries(hp1,delta_t=1./Sfreq,epoch=0)
        hp2=TimeSeries(hp2,delta_t=1./Sfreq,epoch=0)

        hp1.start_time+=St #Matching the starting time of the two files
        hp2.start_time+=St #Matching the starting time of the two files

        TimeCut=TimeTemplate+St

        Sgn1=hp1+Back1
        Sgn2=hp2+Back2
      
        white1 = (Sgn1.to_frequencyseries() / psd1**0.5).to_timeseries()
        white2 = (Sgn2.to_frequencyseries() / psd2**0.5).to_timeseries()

        white1 = resample_to_delta_t(white1, 4.0/white1.sample_rate) #Resample
        white2 = resample_to_delta_t(white2, 4.0/white2.sample_rate) #Resample

        Slice1=white1.time_slice(TimeCut,TimeCut+TimeStep)
        Slice2=white2.time_slice(TimeCut,TimeCut+TimeStep)

        t1, f1, p1 = Slice1.qtransform(.025/2,logfsteps=100,qrange=(8, 8),frange=(FreqRange[0], FreqRange[1]))
        t2, f2, p2 = Slice2.qtransform(.025/2,logfsteps=100,qrange=(8, 8),frange=(FreqRange[0], FreqRange[1]))

        Mean_1=np.mean(p1)
        Std_1=np.std(p1)

        Mean_2=np.mean(p2)
        Std_2=np.std(p2)

        GroupSav[i,:,:,0]=(p1-Mean_1)/Std_1
        GroupSav[i,:,:,1]=(p2-Mean_2)/Std_2

    Hf1.create_dataset('p',data=GroupSav)
    Hf1.create_dataset('SNR',data=SNR)


    Hf1.close()


