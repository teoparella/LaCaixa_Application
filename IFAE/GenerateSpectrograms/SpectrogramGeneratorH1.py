import numpy as np
import pycbc.frame #It's used to read the format of the files: gwf 
import h5py
from pycbc.filter import resample_to_delta_t

Inform='/nfs/virgo/tparella/SandBox/SandBox1/GW/Output2/NoiseOutputH1/H5/Information.h5'
Whiten='/nfs/virgo/tparella/SandBox/SandBox1/GW/Output2/NoiseOutputH1/'
Channel='H1:GWOSC-16KHZ_R1_STRAIN'
Output='/nfs/virgo/tparella/SandBox/SandBox1/GW/Spectrograms/SpectrogramsH1/'
FreqRange=[20,600] #20,512 Nominal one

ImageStep=5. #Segundos de cada imagen
#Careful, if you change the ImageStep and want to maintain the 100x400 you need to change the qtransform delta t parameter
Index='For Condor purposes'

ReadInfo=h5py.File(Inform,'r')
Group=ReadInfo.get(str(Index))

NImages=np.array(Group.get('NTemplates'))
#FileName=Group.attrs.get('Background_file')#####################

print(NImages)

name='%s_SignalWhiten.gwf' % (Index)

Back = pycbc.frame.read_frame(Whiten+name, Channel)
st=float(Back.start_time)

BackWhit=Back.time_slice(st+10,st+4090)
strain = resample_to_delta_t(BackWhit, 4.0/BackWhit.sample_rate) #Resample

h=h5py.File(Output+"%s.h5" % Index,'w')

for n in range(NImages):
    hf=h.create_group(str(n))

    Slice=strain.time_slice(strain.start_time+ImageStep*n,strain.start_time+ImageStep*(n+1))

    t, f, p = Slice.qtransform(.025/2,logfsteps=100,qrange=(8, 8),frange=(FreqRange[0], FreqRange[1]))
    hf.create_dataset('Initial_gps',data=float(Slice.start_time))
    #hf.create_dataset('Original_file',data=FileName)
    hf.create_dataset('p',data=p)
    hf.create_dataset('f',data=f)
    hf.create_dataset('t',data=t)


