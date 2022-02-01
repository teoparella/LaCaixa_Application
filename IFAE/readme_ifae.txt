This repostory includes the work done at IFAE during summer 2020.

The GenerateSpectrograms folder includes all the main programs used for generating theoretical gravitational waves signals with noise. 
SpectrogramGeneratorH1.py and SpectrogramGeneratorL1.py includes theoretical signals for two gravitational interferometer channels. Noisegenerator_SignalSameTimeH5_H1.py and Noisegenerator_SignalSameTimeH5_L1.py include the noise in the theoretical data. Finally, the GW_theoreticaldata_generator.py includes the full generation of the positive set that will feed the upcoming Convolutional Neural Network.

The SNR file, includes SNR.py which elaborates samples of different Signal-to-noise ratio. This is used afterwards to check the CNN performance as function fo the SNR.

The CNN file contains the document resnet50.py. It is an example of CNN, consisting of a resnet.
