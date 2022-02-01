import numpy as np
from pycbc.psd import inverse_spectrum_truncation, interpolate
import pycbc.frame #It's used to read the format of the files: gwf 
import h5py
import sys
from pycbc.waveform import get_td_waveform
from pycbc.filter import matched_filter

BkgH5Path2="/nfs/pic.es/user/a/amenende/H1_O2_Local.h5"
BkgH5Path1="/nfs/pic.es/user/a/amenende/L1_O2_Local.h5"
OutputPath='/nfs/virgo/tparella/SandBox/SandBox1/GW/Output2/NoiseOutputL1/'
InfoPath="/nfs/virgo/tparella/SandBox/SandBox1/GW/Output1/OutputData/Information.h5"
TemplatePath='/nfs/virgo/tparella/SandBox/SandBox1/GW/Output1/OutputL1/'
Channel='L1:DCS-CALIB_STRAIN_C01'

Index='For Condor purposes'

Bkg1=h5py.File(BkgH5Path1,'r')
Bkg2=h5py.File(BkgH5Path2,'r')

name=np.array(Bkg1.get('Files'))
st=np.array(Bkg1.get('Start_times'))
st2=np.array(Bkg2.get('Start_times'))
et=np.array(Bkg1.get('End_times'))
ChannelBkg=str(Bkg1.attrs.get('Channel'))

Duty1=np.array(Bkg1.get('Duty_cycle')).astype(bool)
Duty2=np.array(Bkg2.get('Duty_cycle')).astype(bool)
Events1=np.array(Bkg1.get('Events')).astype(bool)
Events2=np.array(Bkg2.get('Events')).astype(bool)

Filter1=np.logical_and(Duty1,~Events1)
Filter2=np.logical_and(Duty2,~Events2)

name=name[Filter1]
st=st[Filter1]
st2=st2[Filter2]
et=et[Filter1]

st=st.astype(float)
st2=st2.astype(float)

_,Y,_=np.intersect1d(st,st2,return_indices=True)

name=name[Y[-198:]]
st=st[Y[-198:]]
et=et[Y[-198:]]

max_filter_duration=4

hf=h5py.File(InfoPath,'r')
A=name[Index]

background = pycbc.frame.read_frame(A, ChannelBkg)
PSD=background.psd(max_filter_duration)
psd = interpolate(PSD, background.delta_f)
max_filter_len = int(max_filter_duration * background.sample_rate)

psd = inverse_spectrum_truncation(psd,
		   max_filter_len=max_filter_len,
		   low_frequency_cutoff=20,
		   trunc_method='hann')

hp = pycbc.frame.read_frame(TemplatePath+str(Index)+'.gwf', Channel)
hp.start_time+=background.start_time

signal=hp+background

white = (signal.to_frequencyseries() / psd**0.5).to_timeseries()

pycbc.frame.write_frame(OutputPath+str(Index)+"_SignalWhiten.gwf", ChannelBkg, white)

G=hf.get(str(Index))

'''hf2'''

g2=h5py.File(OutputPath+"H5/InformationSignal_"+str(Index)+".h5",'w')

#g2=hf2.create_group(str(Index))

g2.create_dataset('Mass1',data=G.get('Mass1'))
g2.create_dataset('Mass2',data=G.get('Mass2'))
g2.create_dataset('Luminosity_distance',data=G.get('Luminosity_distance'))
g2.create_dataset('hp1_start_time',data=(np.array(G.get('hp1_start_time'))+float(st[Index])))
g2.create_dataset('hp1_end_time',data=(np.array(G.get('hp1_end_time'))+float(st[Index])))
g2.create_dataset('hp2_start_time',data=(np.array(G.get('hp2_start_time'))+float(st[Index])))
g2.create_dataset('hp2_end_time',data=(np.array(G.get('hp2_end_time'))+float(st[Index])))
g2.create_dataset('Beginning_of_interval',data=(np.array(G.get('Beginning_of_interval'))+float(st[Index])))
g2.create_dataset('Interval_lenght',data=G.get('Interval_lenght'))
g2.create_dataset('RA',data=G.get('RA'))
g2.create_dataset('Inclination',data=G.get('Inclination'))
g2.create_dataset('Declination',data=G.get('Declination'))
g2.create_dataset('Polarization',data=G.get('Polarization'))
g2.create_dataset('NTemplates',data=G.get('NTemplates'))
g2.create_dataset('Duration',data=G.get('Duration'))


#g2.attrs['Background_file']='background'

hf.close()
hf2.close()

