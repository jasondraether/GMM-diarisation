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



class GMMClassifier(object):

    def __init__(self, params={}):

        default_params = {
        'train_directory' : '',
        'files_list' : [],
        'classes' : [],
        'sample_rate' : 48000,

        'use_preemphasis' : True,
        'preemphasis_coefficient' : 0.97,

        'normalize_signal' : True,
        'energy_multiplier' : 0.05,
        'energy_range' : 100,

        'n_ccs' : 20,
        'normalize_mfcc' : False,
        'use_deltas' : True,
        'win_len' : 0.5,
        'win_step' : 0.01,

        'n_components' : 3,
        'covariance_type' : 'full',
        'max_iter' : 5000,

        'trim_silence' : True,
        'silence_frame_length' : 512,
        'silence_frame_skip' : 256,
        'top_db' : 30
        }

        # Update default params with user params
        self.params = default_params
        assert type(params) == dict
        for key, value in params.items():
            assert self.params.get(key) != None
            self.params[key] = value

        # Training data parameters
        self.train_directory = self.params.get('train_directory')
        self.files_list = self.params.get('files_list')
        self.classes = self.params.get('classes')
        self.sample_rate = self.params.get('sample_rate')

        # Pre-emphasis filter
        self.use_preemphasis = self.params.get('use_preemphasis')
        self.preemphasis_coefficient = self.params.get('preemphasis_coefficient')

        # Signal normalization
        self.normalize_signal = self.params.get('normalize_signal')
        self.energy_multiplier = self.params.get('energy_multiplier')
        self.energy_range = self.params.get('energy_range')

        # Mel-Frequency Cepstral Coefficient
        self.n_ccs = self.params.get('n_ccs')
        self.normalize_mfcc = self.params.get('normalize_mfcc')
        self.use_deltas = self.params.get('use_deltas')
        self.win_len = self.params.get('win_len')
        self.win_step = self.params.get('win_step')

        # GMM Parameters
        self.n_components = self.params.get('n_components')
        self.covariance_type = self.params.get('covariance_type')
        self.max_iter = self.params.get('max_iter')

        # Silence trimming
        self.trim_silence = self.params.get('trim_silence')
        self.silence_frame_length = self.params.get('silence_frame_length')
        self.silence_frame_skip = self.params.get('silence_frame_skip')
        self.top_db = self.params.get('top_db')

        self.n_classes = len(self.classes)

        if self.train_directory != None:
            assert os.path.exists(self.train_directory)
        elif self.files_list != None:
            pass
        else:
            pass

    def get_params(self):
        return self.params

x = GMMClassifier()
