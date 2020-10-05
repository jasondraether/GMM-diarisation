import numpy as np
import os
from scipy.io import wavfile
from python_speech_features import mfcc
from librosa.feature import delta
from librosa.effects import split
from itertools import chain, combinations, product
from time import sleep, time
from random import shuffle
import logging

class FeatureExtractor(object):

    def __init__(self, params={}):

        default_params = {
        'log_filename' : 'session.log',
        'sample_rate' : 48000, # 32 float PCM

        'use_emphasis' : True,
        'emphasis_coefficient' : 0.97,

        'normalize_signal' : True,
        'energy_multiplier' : 0.05,
        'energy_range' : 100,

        'n_ccs' : 20,
        'normalize_mfcc' : False,
        'max_order' : 1,
        'use_deltas' : True,
        'win_len' : 0.02,
        'win_step' : 0.01,

        'trim_silence' : False,
        'frame_length' : 512,
        'frame_skip' : 256,
        'top_db' : 30
        }

        # Update default params with user params
        self.params = default_params
        self.set_params(params=params)

        # Logger setup
        logging.basicConfig(filename=self.params['log_filename'],level=logging.DEBUG,filemode='w')
        self.logger = logging.getLogger('features')

        # Training data parameters
        self.classes = self.params.get('classes')
        self.sample_rate = self.params.get('sample_rate')

        # Pre-emphasis filter
        self.use_emphasis = self.params.get('use_emphasis')
        self.emphasis_coefficient = self.params.get('emphasis_coefficient')

        # Signal normalization
        self.normalize_signal = self.params.get('normalize_signal')
        self.energy_multiplier = self.params.get('energy_multiplier')
        self.energy_range = self.params.get('energy_range')

        # Mel-Frequency Cepstral Coefficient
        self.n_ccs = self.params.get('n_ccs')
        self.normalize_mfcc = self.params.get('normalize_mfcc')
        self.use_deltas = self.params.get('use_deltas')
        self.max_order = self.params.get('max_order')
        self.win_len = self.params.get('win_len')
        self.win_step = self.params.get('win_step')

        # Silence trimming
        self.trim_silence = self.params.get('trim_silence')
        self.frame_length = self.params.get('frame_length')
        self.frame_skip = self.params.get('frame_skip')
        self.top_db = self.params.get('top_db')

        # Other params initialized
        self.speaker_profiles = {} # Dictionary of GMM models, indexed by class name

    def set_params(self, params={}):
        if type(params) != dict: raise TypeError('Input params must be dictionary.')
        for key, value in params.items():
            if self.params.get(key) == None: raise ValueError('Parameter {0} does not exist within class'.format(key))
            self.params[key] = value

    def get_params(self):
        return self.params

    # From: http://jamesmontgomery.us/blog/Voice_Recognition_Model.html
    def apply_emphasis(self, signal):
        return np.append(signal[0],signal[1:] - self.emphasis_coefficient*signal[:-1])

    # From: http://jamesmontgomery.us/blog/Voice_Recognition_Model.html
    def truncate_silence(self, signal):

        nonsilent_indices = split(y=signal, \
                                  top_db=self.top_db, \
                                  frame_length=self.frame_length, \
                                  hop_length=self.frame_skip)

        # Only keep nonsilent intervals of signal
        signal_intervals = []
        for index in nonsilent_indices:
            signal_interval = signal[index[0]:index[1]]
            signal_intervals.append(signal_interval)

        # Return flattened array
        return np.concatenate(signal_intervals,axis=0)

    # From: http://jamesmontgomery.us/blog/Voice_Recognition_Model.html
    def energy_normalize(self, signal):

        energy = [v**2 for v in signal]
        voiced_threshold = self.energy_multiplier*np.mean(energy)
        clean_samples = [0]

        for sample_set in range(0, len(signal)-self.energy_range, self.energy_range):
            sample_set_th = np.mean(energy[sample_set:sample_set+self.energy_range])
            if sample_set_th > voiced_threshold:
                clean_samples.extend(signal[sample_set:sample_set+self.energy_range])

        return np.array(clean_samples)


    def preprocess(self,signal):
        if self.use_emphasis:
            signal = self.apply_emphasis(signal)
        if self.trim_silence:
            signal = self.truncate_silence(signal)
        if self.normalize_signal:
            signal = self.energy_normalize(signal)

        return signal

    def calculate_mfccs(self,signal):
        # preemph=0 because we use our own pre-emphasis
        mfccs = mfcc(signal=signal, \
                     samplerate=self.sample_rate, \
                     winfunc=np.hamming, \
                     numcep=self.n_ccs, \
                     nfft=int(self.win_len*self.sample_rate), \
                     preemph=0, \
                     winlen=self.win_len, \
                     winstep=self.win_step, \
                     appendEnergy=False) # TODO: investigate appendEnergy?
                     # Maybe don't append energy but tack on an extra
                     # Mfcc or something?

        if self.normalize_mfcc:
            mean = np.mean(mfccs,axis=0)
            std = np.mean(mfccs,axis=0)
            mfccs = (mfccs - mean)/std

        return mfccs

    def calculate_mfcc_deltas(self, mfccs):
        # If order is 2, we want to calculate order=1, and order=2
        n_data = mfccs.shape[0]
        delta_feats = np.zeros((n_data,self.n_ccs*self.max_order))
        for order in range(self.max_order):
            delta_feats[:,order*self.n_ccs:(order+1)*self.n_ccs] = delta(mfccs,order=order+1)
        return delta_feats

    # Given .wav data X, return features array
    def extract_features(self, signal):
        signal_preprocessed = self.preprocess(signal)
        signal_mfccs = self.calculate_mfccs(signal_preprocessed)

        n_data = signal_mfccs.shape[0]

        if self.use_deltas and self.max_order > 0:
            features = np.zeros((n_data,self.n_ccs*(self.max_order+1)))
            features[:,0:self.n_ccs] = signal_mfccs
            signal_deltas = self.calculate_mfcc_deltas(signal_mfccs)
            features[:,self.n_ccs:self.n_ccs*(self.max_order+1)] = signal_deltas
        else:
            features = signal_mfccs
        return features

    # Given .wav file filename (must be full path), return features array
    def extract_features_file(self, filename=''):
        if not os.path.exists(filename): raise ValueError('Filename {0} does not exist.'.format(filename))
        sample_rate, signal = wavfile.read(filename)
        if sample_rate != self.sample_rate: raise ValueError('Sample rate mismatch. {0} =/= {1}'.format(sample_rate,self.sample_rate))
        return self.extract_features(signal)

    # If concatenate, return np array of all features concatenated
    # Otherwise, return list of numpy arrays for each file
    def extract_features_files(self, files=[], concatenate=True):
        all_features = []
        for file in files:
            if not os.path.exists(file): raise ValueError('Filename {0} does not exist.'.format(file))

            file_features = self.extract_features_file(file)
            all_features.append(file_features)

        X = []
        if concatenate:
            for file_features in all_features:
                X.append(file_features)
            X = np.concatenate(X,axis=0)
        else:
            X = all_features
        return X

    # Given directory name, return features array (must be 1-level)
    def extract_features_dir(self, dir='', concatenate=True):
        if not os.path.exists(dir): raise ValueError('Directory {0} does not exist.'.format(dir))
        files_list = os.listdir(dir)
        filenames = []
        for file in files_list:
            filenames.append(os.path.join(dir,file))
        if len(filenames) == 0:
            self.logger.warn('Found 0 files in directory {0}. Skipping.'.format(dir))
        else:
            self.logger.info('Found {0} files in directory {1}.'.format(len(filenames),dir))
            X = self.extract_features_files(filenames,concatenate=concatenate)
        return X
