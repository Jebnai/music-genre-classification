#!/usr/bin/env python
# coding: utf-8

# In[2]:


# sckit-learn
import sklearn

# Librosa
import librosa
import librosa.display
import torch
import IPython
import IPython.display as ipd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# In[13]:


# using 'jazz.00024.wav' to explore audio data
sound, sample_rate = librosa.load('Data/genres_original/jazz/jazz.00015.wav')
IPython.display.Audio(sound, rate=sample_rate)


# In[14]:


# a plot of 2D sound waves for 'jazz.00024.wav'
plt.figure(figsize = (14, 4))
librosa.display.waveshow(sound, sample_rate)
plt.title("'jazz.00015.wav' sound waves", fontsize=24)


# In[15]:


# short-time Fourier transform
transform = np.abs(librosa.stft(sound))
plt.figure()
plt.plot(transform)


# In[16]:


# spectrogram
decibels = librosa.amplitude_to_db(transform)
plt.figure(figsize = (14, 4))
plt.title("'jazz.00015.wav' spectrogram")
librosa.display.specshow(decibels, hop_length = 512, x_axis = 'time', y_axis = 'log')

# mel-spectrogram
scale = librosa.feature.melspectrogram(sound)
plt.figure(figsize = (14,4))
plt.title("'jazz.00015.wav' mel-spectrogram")
librosa.display.specshow(scale, hop_length = 512, x_axis = 'time', y_axis = 'log', cmap='hot')

# mel-frequncy cepstral coefficients (MFCCs)
mfcc = librosa.feature.mfcc(sound)
mfcc = sklearn.preprocessing.scale(mfcc)
plt.figure(figsize=(14, 4))
plt.title("'jazz.00015.wav' MFCC")
librosa.display.specshow(mfcc, x_axis='time')


# In[117]:


# spectral centroid
centroid = librosa.feature.spectral_centroid(sound)[0]
time = librosa.frames_to_time(range(len(centroid)))

# spectral rolloff
rolloff = librosa.feature.spectral_rolloff(sound)[0]
plt.figure(figsize = (14, 4))
plt.title("'jazz.00015.wav' spectral rolloff")
librosa.display.waveshow(sound, alpha=0.4)
plt.plot(time, sklearn.preprocessing.minmax_scale(rolloff, axis=0))


# In[118]:


# music recommender using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

# normalising features
audio_info = pd.read_csv('Data/features_30_sec.csv', index_col='filename')
features = audio_info[['label']]
features.head()
audio_info = audio_info.drop(columns = ['length', 'label'])
audio_info = preprocessing.scale(audio_info)

# cosine_similarity
s = cosine_similarity(audio_info)
sfeatures = pd.DataFrame(s)
snames = sfeatures.set_index(features.index)
snames.columns = features.index

# test example with 'jazz.00024.wav'
list_songs = snames['jazz.00015.wav'].sort_values(ascending = False)
print(list_songs.head(10))  # print the 10 most similar songs


# In[ ]:




