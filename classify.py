import numpy as np
import os
from sklearn.mixture import GaussianMixture as GMM
from scipy.io import wavfile
from librosa.feature import delta
from python_speech_features import mfcc
from librosa.effects import split
import matplotlib.pyplot as plt
from matplotlib import cm
from cv2 import VideoWriter, VideoWriter_fourcc, imread
from itertools import chain, combinations, product
from time import sleep, time
from random import shuffle

# Test parameters randomly
optimize_randomly = False

# Testing Flags
test_standard = False
test_small = False # Just test smaller enumerable parameters, no random shuffling
test_all_parameters = False # This will test every parameter in the model (Warning: LONG)
test_files_and_gmm = False # Just tests file combinations and GMM parameters
test_files_and_parameters = False # This tests all file combinations and ALL parameters in the model (Warning: REALLY LONG)
test_files_only = True # Just tests all file combinations
test_preprocessing = False # Only test possible preprocessing steps
n_files = 4 # Number of files to test, TOTAL (so, if 2 is put here, 1 for matt and 1 for ryan,
# but not guaranteed to be evenly distributed, if 4 the split could be 3 and 1, but class will always have at least one file
use_n_files = True # Whether or not to apply the n_files check

# Caches to speed up testing
cache = []
cache_parameters = []
cache_limit = 16

class GMMClassifier:

    # Optimal parameters from testing should be placed in here
    def __init__(self, data_directory='', from_directory=True, specify_files=False, files_list=[], classes=[],use_emphasis=True, normalize_signal=True, \
    normalize_mfcc=False, use_deltas=True, trim_silence=True, use_ubm=False, ubm_directory='', \
    pad_silence=False, n_ccs=13, n_components=4, covariance_type='tied',win_len=0.025, \
    win_step=0.01, frame_length=512, frame_skip=256, top_db=30):

        # Model information
        self.speaker_profiles = []
        self.classes = []
        self.n_classes = 0
        self.train_corpus_dict = {}
        self.sample_rate = 48000

        # Signal preprocessing
        self.use_emphasis = use_emphasis # Pre-emphasis filter
        self.emphasis_coefficient = 0.97

        self.normalize_signal = normalize_signal # Signal energy normalize
        self.energy_multiplier = 0.05
        self.energy_range = 100

        # MFCC's
        self.n_ccs = n_ccs
        self.epsilon = 1e-10
        self.normalize_mfcc = normalize_mfcc
        self.use_deltas = use_deltas
        self.win_len = win_len
        self.win_step = win_step

        # GMM
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.use_ubm = use_ubm
        self.n_ubm_components = 2048
        self.ubm_covariance_type = 'tied'
        self.decision_threshold = 0 # Adding this in to be used with ubm eventually

        # Trimming silence
        self.frame_length = frame_length
        self.frame_skip = frame_skip
        self.top_db = top_db
        self.trim_silence = trim_silence
        self.pad_silence = pad_silence # Pad silence (zeros) on each end
        self.pad_length = 100 # This number in ms for each side

        # Debugging (Makes a bunch of graphs pop up)
        self.graph = False

        # Initialize model from directory
        if from_directory:
            assert os.path.exists(data_directory)
            classes = os.listdir(data_directory)
            n_classes = len(classes)
            for class_id in range(n_classes):
                class_training_files = []
                class_directory = os.path.join(data_directory,classes[class_id])
                for filename in os.listdir(class_directory):
                    training_path = os.path.join(class_directory,filename)
                    class_training_files.append(training_path)
                self.add_profile(label=classes[class_id],files=class_training_files)

        if use_ubm:
            assert os.path.exists(ubm_directory)
            ubm_training_files = []
            ubm_files = os.listdir(ubm_directory)
            for filename in ubm_files:
                ubm_path = os.path.join(ubm_directory,filename)
                ubm_training_files.append(ubm_path)
            self.add_profile(label='ubm',files=ubm_training_files)

        self.current_params = ['use_emphasis: {0}'.format(use_emphasis), \
                          'normalize_signal: {0}'.format(normalize_signal), \
                          'normalize_mfcc: {0}'.format(normalize_mfcc), \
                          'use_deltas: {0}'.format(use_deltas), \
                          'trim_silence: {0}'.format(trim_silence), \
                          'use_ubm: {0}'.format(use_ubm), \
                          'n_ccs: {0}'.format(n_ccs), \
                          'covariance_type: {0}'.format(covariance_type), \
                          'n_components: {0}'.format(n_components), \
                          'win_len: {0}'.format(win_len), \
                          'win_step: {0}'.format(win_step), \
                          'frame_length: {0}'.format(frame_length), \
                          'frame_skip: {0}'.format(frame_skip), \
                          'top_db: {0}'.format(top_db)]

    def add_profile(self, label='', files=''):

        # So we can use this globally
        global cache
        global cache_limit
        global cache_parameters

        # Initialize training corpus dictionary
        self.train_corpus_dict[label] = 0

        assert len(files) > 0

        if label == 'ubm':
            gmm = GMM(n_components=self.n_ubm_components,covariance_type=self.ubm_covariance_type)
        else:
            gmm = GMM(n_components=self.n_components,covariance_type=self.covariance_type)

        if test_small or test_files_only or test_files_and_gmm:
            if len(cache) > cache_limit: # Clear cache when too full
                cache = []
                cache_parameters = []

        feature_list = []
        for filepath in files:
            assert os.path.exists(filepath)

            self.caching_params = [self.use_emphasis,self.normalize_signal,self.normalize_mfcc,self.use_deltas,self.trim_silence,self.n_ccs,filepath] # ONLY currently usable for test_small

            if test_small or test_files_only or test_files_and_gmm:

                if self.caching_params in cache_parameters:
                        cache_index = cache_parameters.index(self.caching_params)
                        features = cache[cache_index]
                        #print("Cache hit: {0}".format(self.caching_params))

                else:
                    # Get audio data from filepath
                    sample_rate, signal = wavfile.read(filepath)
                    assert sample_rate == self.sample_rate

                    # Prepare signal and features
                    signal = self.preprocess_signal(signal)
                    features = self.calculate_features(signal=signal,sample_rate=sample_rate)

                    cache_parameters.append(self.caching_params)
                    cache.append(features)
                    #print("Added new cache entry for params: {0}".format(self.caching_params))


            else:
                # Get audio data from filepath
                sample_rate, signal = wavfile.read(filepath)
                assert sample_rate == self.sample_rate

                #print("Calculating features from {0}".format(filepath))
                # Prepare signal and features
                signal = self.preprocess_signal(signal)
                features = self.calculate_features(signal=signal,sample_rate=sample_rate)

            feature_list.append(features)

        x_train = np.concatenate(feature_list,axis=0)

        # Fit GMM and update training corpus dictionary
        #print("Fitting GMM for {0} with {1} data points".format(label,x_train.shape[0]))
        gmm.fit(x_train)
        self.train_corpus_dict[label] += x_train.shape[0]

        # Append new trained GMM and update class metadata
        if label == 'ubm':
            self.ubm = gmm
        else:
            self.speaker_profiles.append(gmm)
            self.classes.append(label)
            self.n_classes += 1

    # From: http://jamesmontgomery.us/blog/Voice_Recognition_Model.html
    def plot_mfccs(self, mfcc):
        fig, ax = plt.subplots(figsize=(20,5))
        cax = ax.imshow(mfcc.T, interpolation='nearest', cmap=cm.RdYlBu, origin='lower', aspect='auto')
        plt.title("MFCCs")
        plt.xlabel('Time (Sec)')
        plt.ylabel('Frequency (kHz)')
        a = np.array( ax.get_xticks().tolist() )
        a = a / 100.0
        ax.set_xticklabels(a)
        plt.show()

    # From: http://jamesmontgomery.us/blog/Voice_Recognition_Model.html
    def plot_signal(self, signal, sample_rate, title='Signal'):
        Time=np.linspace(0, len(signal)/sample_rate, num=len(signal))
        plt.figure(figsize=(20,5))
        plt.plot(Time,signal)
        plt.title(title)
        plt.ylabel('Amplitude')
        plt.xlabel('Time (Sec)')
        plt.show()

    # From: http://jamesmontgomery.us/blog/Voice_Recognition_Model.html
    def energy_normalization(self, signal):

        energy = [a**2 for a in signal]
        voiced_threshold = self.energy_multiplier*np.mean(energy)
        clean_samples = [0]

        for sample_set in range(0, len(signal)-self.energy_range, self.energy_range):
            sample_set_th = np.mean(energy[sample_set:sample_set+self.energy_range])
            if sample_set_th > voiced_threshold:
                clean_samples.extend(signal[sample_set:sample_set+self.energy_range])

        return np.array(clean_samples)

    # (Some) From: http://jamesmontgomery.us/blog/Voice_Recognition_Model.html
    def clear_silence(self, signal):

        nonsilent_indices = split(y=signal,top_db=self.top_db, \
                        frame_length=self.frame_length,hop_length=self.frame_skip)

        # Only keep nonsilent intervals of signal
        signal_intervals = []
        for index in nonsilent_indices:
            signal_interval = signal[index[0]:index[1]]

            if self.pad_silence: # Pad both ends with zeros
                n_samples = int((self.sample_rate*self.pad_length)/1000) # Number of samples to pad
                signal_length = signal_interval.shape[0]
                padded_signal = np.zeros((signal_length+(2*n_samples)))
                padded_signal[n_samples:signal_length+n_samples] = signal_interval
                signal_interval = padded_signal

            signal_intervals.append(signal_interval)

        # Return flattened array
        return np.concatenate(signal_intervals,axis=0)

    def calculate_features(self, signal, sample_rate):

        # We disable python_speech_features's pre-emphasis filter
        mfcc_feats = mfcc(signal,samplerate=sample_rate,winfunc=np.hamming, \
        numcep=self.n_ccs,nfft=4096,preemph=0,winlen=self.win_len, \
        winstep=self.win_step) # TODO: See if librosa's MFCC is better

        if self.normalize_mfcc:
            mean = np.mean(mfcc_feats,axis=0)
            std = np.std(mfcc_feats,axis=0)

            mfcc_feats = mfcc_feats - mean

            # mfcc_feats = (mfcc_feats - mean)/(std + self.epsilon)
            # Above is another way of normalization, but doesn't seem
            # to work well in practice. Epsilon is included to avoid
            # dividing by zero

        # These deltas are approximations
        if self.use_deltas:
            delta_feats = delta(mfcc_feats,order=1)
            delta2_feats = delta(mfcc_feats,order=2)

            features = np.zeros((mfcc_feats.shape[0],(self.n_ccs*3)))
            features[:,0:self.n_ccs] = mfcc_feats
            features[:,self.n_ccs:2*self.n_ccs] = delta_feats
            features[:,2*self.n_ccs:3*self.n_ccs] = delta2_feats
        else:
            features = mfcc_feats

        return features

    # From: http://jamesmontgomery.us/blog/Voice_Recognition_Model.html
    def apply_preemphasis(self, signal):
        return np.append(signal[0],signal[1:] - self.emphasis_coefficient*signal[:-1])

    # Wraps all preprocessing helper functions into one
    def preprocess_signal(self, signal):

        if self.graph:
            self.plot_signal(signal,sample_rate,'Starting Signal')

        if self.use_emphasis:
            signal = self.apply_preemphasis(signal)
            if self.graph:
                self.plot_signal(signal,sample_rate,'Emphasized Signal')

        if self.trim_silence:
            signal = self.clear_silence(signal)
            if self.graph:
                self.plot_signal(signal,sample_rate,'Silenced Signal')

        if self.normalize_signal:
            signal = self.energy_normalization(signal)
            if self.graph:
                self.plot_signal(signal,sample_rate,'Normalized Signal')

        if self.graph:
            self.plot_signal(signal,sample_rate,'Final Signal')

        return signal


    def predict_from_file(self, filepath=''):
        assert os.path.exists(filepath)

        sample_rate, signal = wavfile.read(filepath)

        return self.predict_from_array(signal=signal,sample_rate=sample_rate)

    def predict_from_array(self, signal, sample_rate):

        assert sample_rate == self.sample_rate

        signal = self.preprocess_signal(signal)
        x_test = self.calculate_features(signal=signal, sample_rate=sample_rate)

        log_likelihood = np.zeros((x_test.shape[0],self.n_classes))

        if self.use_ubm:
            ubm_llh = self.ubm.score_samples(x_test)

        for class_id in range(self.n_classes):
            gmm = self.speaker_profiles[class_id]
            llh = gmm.score_samples(x_test)
            if self.use_ubm:
                log_likelihood[:,class_id] = llh/ubm_llh
            else:
                log_likelihood[:,class_id] = llh

        return log_likelihood

    def evaluate_model(self, test_directory=''):

        assert os.path.exists(test_directory)

        tests = []
        test_corpus_dict = {}

        test_classes = os.listdir(test_directory)
        for test_class in test_classes:
            assert test_class in self.classes
            test_corpus_dict[test_class] = 0
            test_class_directory = os.path.join(test_directory,test_class)
            test_class_files = os.listdir(test_class_directory)
            for test_file in test_class_files:
                test_filepath = os.path.join(test_class_directory,test_file)
                tests.append([test_filepath,test_class])

        n_correct = 0
        n_tests = 0

        for test in tests:
            llh = self.predict_from_file(test[0])

            n_data = llh.shape[0]
            assert n_data > 0

            predictions = np.argmax(llh,axis=1)

            # Definitely a way to optimize this but this is python
            for p in predictions:
                if self.classes[p] == test[1]:
                    n_correct += 1

            test_corpus_dict[test[1]] += n_data
            n_tests += n_data

        accuracy = n_correct / n_tests

        return accuracy, test_corpus_dict

    # Various getters to help in main
    def get_classes(self):
        return self.classes

    def get_train_corpus_dict(self):
        return self.train_corpus_dict

    def get_sample_rate(self):
        return self.sample_rate

    def get_params(self):
        return self.current_params

# Helper functions for main body
def powerset(iterable):
    s = list(iterable)
    # Excludes empty set
    pset = list(chain.from_iterable(combinations(s,r) for r in range(1,len(s)+1)))
    # Turn into a list of lists (default list conversion has a bunch of tuples)
    pset_list = []
    for set in pset:
        subset = []
        for element in set:
            subset.append(element)
        pset_list.append(subset)
    return pset_list

def cartesian_product(iterable):
    c_product = list(product(*iterable))

    product_list = []
    for p in c_product:
        sublist = []
        for element in p:
            sublist.append(element)
        product_list.append(sublist)
    return product_list

# Flattens our cartesian product (really not well written)
def flatten_cartesian(cartesian):
    flat_cartesian = []
    for sublist in cartesian:
        for file in sublist:
            flat_cartesian.append(file)

    return flat_cartesian

def parse_files(file_list, class_id):
    parsed_files = []
    for f in file_list:
        if f[1] == class_id:
            parsed_files.append(f[0])
    assert len(parsed_files) > 0
    return parsed_files

def keep_best(best_accuracy,best_params,current_accuracy,current_params):
    if current_accuracy > best_accuracy:
        print("New best:",current_accuracy,current_params)
        return current_accuracy, current_params
    else:
        return best_accuracy, best_params

def main():

    # TODO: Make this smaller and make it easier to test params without explicitly listing them!!!

    # Data directories for training and testing
    data_directory = 'profile_data/'
    test_directory = 'test_data/'
    ubm_directory = 'ubm_data/'

    # Performance Flags
    animate = False

    # File override
    specify_files = False # TODO: Add more support for this in the class structure itself
    classes = ['matt','ryan'] # Match up class_id and class in file_list based on position
    file_list = []
    n_classes = len(classes)

    # Enumerable parameters
    _use_emphasis = [True, False]
    _normalize_signal = [True, False]
    _normalize_mfcc = [True, False]
    _use_deltas = [True, False]
    _trim_silence = [True, False]
    _use_ubm = [True, False]
    _n_ccs = [13, 20]
    _covariance_type = ['full','tied','diag','spherical']

    # Parameters boundaries
    min_components = 1
    max_components = 32
    win_len_coeff = 0.005
    min_win_len = int(0.01/win_len_coeff)
    max_win_len = int(0.05/win_len_coeff)
    win_step_coeff = 0.0025
    min_win_step = int(0.005/win_step_coeff)
    max_win_step = int(0.03/win_step_coeff)
    min_frame_length = 7 # Powers of 2
    max_frame_length = 13
    min_frame_skip = 5 # Powers of 2
    max_frame_skip = 11
    min_top_db = 20
    max_top_db = 40
    top_db_inc = 5

    # Ranges for parameters
    _n_components = [i for i in range(min_components,max_components+1)]
    _win_len = [win_len_coeff*i for i in range(min_win_len,max_win_len+1)]
    _win_step = [win_step_coeff*i for i in range(min_win_step,max_win_step+1)]
    _frame_length = [2**i for i in range(min_frame_length,max_frame_length+1)]
    _frame_skip = [2**i for i in range(min_frame_skip,max_frame_skip+1)]
    _top_db = [i for i in range(min_top_db,max_top_db+top_db_inc,top_db_inc)]

    best_accuracy = 0.0
    best_params = []

    # Test current parameters
    if test_standard:

        s = time()
        classifier = GMMClassifier(data_directory=data_directory,from_directory=True,ubm_directory=ubm_directory)
        e = time()
        print("Training time in test_standard: {0}".format(e-s))
        s = time()
        current_accuracy, test_corpus_dict = classifier.evaluate_model(test_directory=test_directory)
        e = time()
        print("Evaluation time in test_standard: {0}".format(e-s))
        current_params = classifier.get_params()
        best_accuracy = current_accuracy
        best_params = current_params
        #print(current_accuracy, current_params)

    elif test_preprocessing: # TODO: Include n_ccs in this?
        parameters_list = list(product(_use_emphasis, \
                                          _normalize_signal, \
                                          _normalize_mfcc, \
                                          _use_deltas, \
                                          _trim_silence, \
                                          _n_ccs \
                                          ))
        for use_emphasis, \
            normalize_signal, \
            normalize_mfcc, \
            use_deltas, \
            trim_silence, \
            n_ccs in parameters_list:

            current_params = ['use_emphasis: {0}'.format(use_emphasis), \
                              'normalize_signal: {0}'.format(normalize_signal), \
                              'normalize_mfcc: {0}'.format(normalize_mfcc), \
                              'use_deltas: {0}'.format(use_deltas), \
                              'trim_silence: {0}'.format(trim_silence), \
                              'n_ccs: {0}'.format(n_ccs)]

            classifier = GMMClassifier(data_directory=data_directory, \
                                       from_directory=True, \
                                       use_emphasis=use_emphasis, \
                                       normalize_signal=normalize_signal, \
                                       normalize_mfcc=normalize_mfcc, \
                                       use_deltas=use_deltas, \
                                       trim_silence=trim_silence, \
                                       n_ccs=n_ccs)

            current_accuracy, test_corpus_dict = classifier.evaluate_model(test_directory=test_directory)
            best_accuracy, best_params = keep_best(best_accuracy,best_params, current_accuracy, current_params)
            print(current_accuracy, current_params)

    elif test_small:
        parameters_list = list(product(_use_emphasis, \
                                          _normalize_signal, \
                                          _normalize_mfcc, \
                                          _use_deltas, \
                                          _trim_silence, \
                                          _use_ubm, \
                                          _n_ccs, \
                                          _covariance_type \
                                          ))

        for use_emphasis, \
            normalize_signal, \
            normalize_mfcc, \
            use_deltas, \
            trim_silence, \
            use_ubm, \
            n_ccs, \
            covariance_type in parameters_list:
            for n_components in _n_components:

                current_params = ['use_emphasis: {0}'.format(use_emphasis), \
                                  'normalize_signal: {0}'.format(normalize_signal), \
                                  'normalize_mfcc: {0}'.format(normalize_mfcc), \
                                  'use_deltas: {0}'.format(use_deltas), \
                                  'trim_silence: {0}'.format(trim_silence), \
                                  'use_ubm: {0}'.format(use_ubm), \
                                  'n_ccs: {0}'.format(n_ccs), \
                                  'covariance_type: {0}'.format(covariance_type), \
                                  'n_components: {0}'.format(n_components)]

                classifier = GMMClassifier(data_directory=data_directory, \
                                           from_directory=True, \
                                           use_emphasis=use_emphasis, \
                                           normalize_signal=normalize_signal, \
                                           normalize_mfcc=normalize_mfcc, \
                                           use_deltas=use_deltas, \
                                           trim_silence=trim_silence, \
                                           use_ubm=use_ubm, \
                                           ubm_directory=ubm_directory, \
                                           n_ccs=n_ccs, \
                                           n_components=n_components, \
                                           covariance_type=covariance_type)

                current_accuracy, test_corpus_dict = classifier.evaluate_model(test_directory=test_directory)
                best_accuracy, best_params = keep_best(best_accuracy,best_params, current_accuracy, current_params)
                print(current_accuracy, current_params)

    elif test_all_parameters:
        # A list of all the manageable parameters
        parameters_list = list(product(_use_emphasis, \
                                          _normalize_signal, \
                                          _normalize_mfcc, \
                                          _use_deltas, \
                                          _trim_silence, \
                                          _use_ubm, \
                                          _n_ccs, \
                                          _covariance_type \
                                          ))


        if optimize_randomly:
            shuffle(parameters_list)
            shuffle(_n_components)
            shuffle(_win_len)
            shuffle(_win_step)
            shuffle(_frame_length)
            shuffle(_frame_skip)
            shuffle(_top_db)

        # Tune all hyperparameters
        for use_emphasis, \
            normalize_signal, \
            normalize_mfcc, \
            use_deltas, \
            trim_silence, \
            use_ubm, \
            n_ccs, \
            covariance_type in parameters_list:

            for n_components in _n_components:
                for win_len in _win_len:
                    for win_step in _win_step:
                        for frame_length in _frame_length:
                            for frame_skip in _frame_skip:
                                for top_db in _top_db:
                                    current_params = ['use_emphasis: {0}'.format(use_emphasis), \
                                                      'normalize_signal: {0}'.format(normalize_signal), \
                                                      'normalize_mfcc: {0}'.format(normalize_mfcc), \
                                                      'use_deltas: {0}'.format(use_deltas), \
                                                      'trim_silence: {0}'.format(trim_silence), \
                                                      'use_ubm: {0}'.format(use_ubm), \
                                                      'n_ccs: {0}'.format(n_ccs), \
                                                      'covariance_type: {0}'.format(covariance_type), \
                                                      'n_components: {0}'.format(n_components), \
                                                      'win_len: {0}'.format(win_len), \
                                                      'win_step: {0}'.format(win_step), \
                                                      'frame_length: {0}'.format(frame_length), \
                                                      'frame_skip: {0}'.format(frame_skip), \
                                                      'top_db: {0}'.format(top_db)]

                                    classifier = GMMClassifier(data_directory=data_directory, \
                                                               from_directory=True, \
                                                               use_emphasis=use_emphasis, \
                                                               normalize_signal=normalize_signal, \
                                                               normalize_mfcc=normalize_mfcc, \
                                                               use_deltas=use_deltas, \
                                                               trim_silence=trim_silence, \
                                                               use_ubm=use_ubm, \
                                                               ubm_directory=ubm_directory, \
                                                               n_ccs=n_ccs, \
                                                               n_components=n_components, \
                                                               covariance_type=covariance_type, \
                                                               win_len=win_len, \
                                                               win_step=win_step, \
                                                               frame_length=frame_length, \
                                                               frame_skip=frame_skip, \
                                                               top_db=top_db)

                                    current_accuracy, test_corpus_dict = classifier.evaluate_model(test_directory=test_directory)
                                    best_accuracy, best_params = keep_best(best_accuracy,best_params, current_accuracy, current_params)
                                    #print(current_accuracy, current_params)
    else:
        # Generate file combinations (this kind of overrides the stuff
        # that the class does, but it's mainly for optimization)

        file_powersets = []
        assert os.path.exists(data_directory)
        classes = os.listdir(data_directory)
        n_classes = len(classes)
        for class_id in range(n_classes):
            training_filenames = []
            class_directory = os.path.join(data_directory,classes[class_id])
            class_files = os.listdir(class_directory)
            for class_file in class_files:
                filepath = os.path.join(class_directory,class_file)
                training_filenames.append([filepath, class_id])

            training_powerset = powerset(training_filenames)
            file_powersets.append(training_powerset)

        powerset_product = cartesian_product(file_powersets)

        if optimize_randomly:
            shuffle(powerset_product)

        if test_files_and_gmm:

            if optimize_randomly:
                shuffle(_n_components)
                shuffle(_covariance_type)

            for file_combination in powerset_product:
                file_combination_flat = flatten_cartesian(file_combination)
                if use_n_files and len(file_combination_flat) == n_files: # If TOTAL number of files (i.e., across all classes) exceeds n_files
                    continue
                for n_components in _n_components:
                    for covariance_type in _covariance_type:
                        current_params = ['file_combination: {0}'.format(file_combination_flat), \
                                          'n_components: {0}'.format(n_components), \
                                          'covariance_type: {0}'.format(covariance_type)]

                        classifier = GMMClassifier(from_directory=False, \
                                                   n_components=n_components, \
                                                   covariance_type=covariance_type, \
                                                   ubm_directory=ubm_directory)

                        for class_id in range(n_classes):
                            training_filenames = parse_files(file_combination_flat,class_id)
                            classifier.add_profile(label=classes[class_id],files=training_filenames)

                        current_accuracy, test_corpus_dict = classifier.evaluate_model(test_directory=test_directory)
                        best_accuracy, best_params = keep_best(best_accuracy,best_params, current_accuracy, current_params)
                        print(current_accuracy, current_params)

        elif test_files_only:

            for file_combination in powerset_product:
                file_combination_flat = flatten_cartesian(file_combination)
                if use_n_files and len(file_combination_flat) != n_files:
                    continue
                current_params = file_combination_flat

                classifier = GMMClassifier(from_directory=False,ubm_directory=ubm_directory)

                for class_id in range(n_classes):
                    training_filenames = parse_files(file_combination_flat,class_id)
                    classifier.add_profile(label=classes[class_id],files=training_filenames)

                current_accuracy, test_corpus_dict = classifier.evaluate_model(test_directory=test_directory)
                best_accuracy, best_params = keep_best(best_accuracy, best_params, current_accuracy, current_params)
                print(current_accuracy, current_params)

        elif test_files_and_parameters: # The big test
            # A list of all the manageable parameters
            parameters_list = list(product(_use_emphasis, \
                                              _normalize_signal, \
                                              _normalize_mfcc, \
                                              _use_deltas, \
                                              _trim_silence, \
                                              _use_ubm, \
                                              _n_ccs, \
                                              _covariance_type \
                                              ))

            if optimize_randomly:
                shuffle(parameters_list)
                shuffle(_n_components)
                shuffle(_win_len)
                shuffle(_win_step)
                shuffle(_frame_length)
                shuffle(_frame_skip)
                shuffle(_top_db)

            for file_combination in powerset_product:
                file_combination_flat = flatten_cartesian(file_combination)
                if use_n_files and len(file_combination_flat) > n_files:
                    continue
                for use_emphasis, \
                    normalize_signal, \
                    normalize_mfcc, \
                    use_deltas, \
                    trim_silence, \
                    use_ubm, \
                    n_ccs, \
                    covariance_type in parameters_list:

                    for n_components in _n_components:
                        for win_len in _win_len:
                            for win_step in _win_step:
                                for frame_length in _frame_length:
                                    for frame_skip in _frame_skip:
                                        for top_db in _top_db:
                                            current_params = ['file_combination: {0}'.format(file_combination_flat), \
                                                              'use_emphasis: {0}'.format(use_emphasis), \
                                                              'normalize_signal: {0}'.format(normalize_signal), \
                                                              'normalize_mfcc: {0}'.format(normalize_mfcc), \
                                                              'use_deltas: {0}'.format(use_deltas), \
                                                              'trim_silence: {0}'.format(trim_silence), \
                                                              'use_ubm: {0}'.format(use_ubm), \
                                                              'n_ccs: {0}'.format(n_ccs), \
                                                              'covariance_type: {0}'.format(covariance_type), \
                                                              'n_components: {0}'.format(n_components), \
                                                              'win_len: {0}'.format(win_len), \
                                                              'win_step: {0}'.format(win_step), \
                                                              'frame_length: {0}'.format(frame_length), \
                                                              'frame_skip: {0}'.format(frame_skip), \
                                                              'top_db: {0}'.format(top_db)]

                                            classifier = GMMClassifier(from_directory=False, \
                                                                       use_emphasis=use_emphasis, \
                                                                       normalize_signal=normalize_signal, \
                                                                       normalize_mfcc=normalize_mfcc, \
                                                                       use_deltas=use_deltas, \
                                                                       trim_silence=trim_silence, \
                                                                       use_ubm=use_ubm, \
                                                                       ubm_directory=ubm_directory, \
                                                                       pad_silence=pad_silence, \
                                                                       n_ccs=n_ccs, \
                                                                       n_components=n_components, \
                                                                       covariance_type=covariance_type, \
                                                                       win_len=win_len, \
                                                                       win_step=win_step, \
                                                                       frame_length=frame_length, \
                                                                       frame_skip=frame_skip, \
                                                                       top_db=top_db)

                                            for class_id in range(n_classes):
                                                training_filenames = parse_files(file_combination_flat,class_id)
                                                classifier.add_profile(label=classes[class_id],files=training_filenames)

                                            current_accuracy, test_corpus_dict = classifier.evaluate_model(test_directory=test_directory)
                                            best_accuracy, best_params = keep_best(best_accuracy,best_params, current_accuracy, current_params)
                                            #print(current_accuracy, current_params)

    print("Best accuracy: {0} \nBest parameters {1}:".format(best_accuracy,best_params))

    if animate:
        # Assumes optimal parameters are defaults
        if specify_files:
            classifier = GMMClassifier(from_directory=False, ubm_directory=ubm_directory)
            for class_id in range(n_classes):
                training_filenames = parse_files(file_list,class_id)
                classifier.add_profile(label=classes[class_id],files=training_filenames)

        else:
            classifier = GMMClassifier(data_directory=data_directory,from_directory=True, ubm_directory=ubm_directory)

        target_podcast = 'unknown_data/196.wav' # Not going to generalize this, make it a specific podcast
        animation_directory = 'animation_data/'

        animation_filename = 'animation.mp4'
        video_width = 1920
        video_height = 1080
        FPS = 60
        max_seconds = 100 # Animation will stop after max_seconds seconds
        interval_length = 0.5 # How many second intervals to slice podcast into

        fourcc = VideoWriter_fourcc(*'mp4v')
        video = VideoWriter(video_filename, fourcc, float(FPS), (width,height))

        still_filename = 'still.png'
        still_frame = imread(os.path.join(animation_directory, still_filename))

        class_frames = []

        for class_id in range(len(classes)):
            class_frame_filename = os.path.join(animation_directory,classes[class_id]+'.png')
            frame = imread(class_frame_filename)
            class_frames.append(frame)

        sample_rate, podcast = wavfile.read(target_podcast)
        n_samples = podcast.shape[0]
        n_seconds = n_samples//sample_rate
        frames_per_prediction = int(interval_length*FPS)

        animation_upper_range = (frames_per_prediction*2)//3
        animation_range_lower = (frames_per_prediction)//3

        for start in range(0,n_samples-sample_length,sample_length):
            llh = classifer.predict_from_array(signal=podcast[start:start+sample_length],sample_rate=sample_rate)
            prediction = np.argmax(np.mean(llh,axis=0))

            for f in range(frames_per_prediction):
                if f > animation_range_lower and f < animation_range_upper:
                    video.write(still_frame)
                else:
                    video.write(class_frames[prediction])

            timestamp = start/sample_rate
            print("Wrote timestamp {0}".format(timestamp))

            if timestamp >= max_seconds:
                print("Exceeded max seconds")
                break

        video.release()

if __name__ == '__main__':
    main()
