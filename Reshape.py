import pandas as pd
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import os
import gc

gc.enable()
df = pd.read_csv("df2.csv")
for i, row in df.iterrows():
    path = row['filename']
    try:
        sound= np.zeros((32000*6))
        sf.read("train_audio/"+path,out=sound) #Path to sound file from df2.csv
        spectrogram , l ,g,r = plt.specgram(sound,Fs=32000)
    except:    
        sound= np.zeros((32000*6,2))
        sf.read("train_audio/"+path,out=sound)
        spectrogram , l ,g,r = plt.specgram(sound[:,1],Fs=32000)
    del sound
    del l,g,r
    gc.collect()
    spectrogram = np.resize(spectrogram,(1,128,128,1)).astype(np.float32)
    patharr = path.split("/")
    if(os.path.exists("train_spec_128/")!=True):
        os.mkdir("train_spec_128/")
    if(os.path.exists("train_spec_128/"+patharr[0])!=True):
        os.mkdir("train_spec_128/"+patharr[0])
    np.save("train_spec_128\\"+patharr[0]+"\\"+patharr[1][:-4]+".npy",spectrogram)
    del spectrogram
    print("Completed " +str(i)+" of "+str(len(df)))