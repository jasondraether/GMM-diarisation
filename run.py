import numpy as np
import os
from sklearn.mixture import GaussianMixture as GMM
from scipy.io import wavfile
from python_speech_features import mfcc
from librosa.effects import split
import matplotlib.pyplot as plt
from matplotlib import cm
from cv2 import VideoWriter, VideoWriter_fourcc, imread
from itertools import chain, combinations, product
from time import sleep, time
from random import shuffle
import logging

# Local modules
from model import SpeakerDiarizer
from features import FeatureExtractor

# Instantiate model and feature extractor
d_params = {}
diarizer = SpeakerDiarizer(d_params)

f_params = {}
extractor = FeatureExtractor(f_params)

# Tune hyperparameters with cross-validation
data_directory = 'profile_data/'
classes = os.listdir(data_directory)
diarizer.init_profiles(labels=classes)
X_train = []
y_train = []
D = len(classes)
# This only works when we concatenate data, if we don't we have to do a little extra
for label in classes:
    class_dir = os.path.join(data_directory,label)
    X_class = extractor.extract_features_dir(dir=class_dir, concatenate=True)
    N = X_class.shape[0]
    y_class = diarizer.label_to_vector(label=label,N=N,D=D)
    X_train.append(X_class)
    y_train.append(y_class)

X_train = np.concatenate(X_train,axis=0)
y_train = np.concatenate(y_train,axis=0)

accuracies = diarizer.cross_validate(X=X_train,y=y_train,n_folds=5)
print(accuracies)
