import numpy as np
import os
from sklearn.mixture import GaussianMixture as GMM
from itertools import chain, combinations, product
from time import sleep, time
from random import shuffle
import logging

class SpeakerDiarizer(object):

    def __init__(self, params={}):

        default_params = {
        'log_filename' : 'session.log',

        'n_components' : 3,
        'covariance_type' : 'full',
        'max_iter' : 5000,
        }

        # Update default params with user params
        self.params = default_params
        assert type(params) == dict
        for key, value in params.items():
            try:
                assert self.params.get(key) != None # Make sure it is a valid param
            except AssertionError as e:
                e.args += ('Parameter {0} does not exist within class.'.format(key))
                raise
            self.params[key] = value

        # Logger setup
        logging.basicConfig(filename=self.params['log_filename'],level=logging.DEBUG,filemode='a')
        self.logger = logging.getLogger('model')

        # GMM Parameters
        self.n_components = self.params.get('n_components')
        self.covariance_type = self.params.get('covariance_type')
        self.max_iter = self.params.get('max_iter')

        # Other params initialized
        self.speaker_profiles = {} # Dictionary of GMM models, indexed by class name
        self.speaker_ids = {} # Map each speaker to a unique ID
        self.n_speakers = 0 # Used to map each speaker
        self.speaker_labels = [] # Label of speaker

    def get_params(self):
        return self.params

    def init_profiles(self, labels):
        for label in labels:
            if self.speaker_profiles.get(label) == None:
                self.logger.info('Creating new model for class'.format(label))
                new_profile = GMM(n_components=self.n_components, \
                                  covariance_type=self.covariance_type, \
                                  max_iter=self.max_iter)
                self.speaker_profiles[label] = new_profile
                self.speaker_ids[label] = self.n_speakers
                self.n_speakers += 1
                self.speaker_labels.append(label)
            else:
                self.logger.warn('Tried to create new model for {0}, but model already exists.'.format(label))

    # Remove all profiles and metadata in object
    def clear_profiles(self):
        self.speaker_profiles = {}
        self.speaker_ids = {}
        self.n_speakers = 0
        self.speaker_labels = []

    # Keep profiles, but clear their GMM's
    def reset_profiles(self):
        for label in self.speaker_labels:
            self.speaker_profiles[label] = GMM(n_components=self.n_components, \
                                               covariance_type=self.covariance_type, \
                                               max_iter=self.max_iter)

    # Fit profile based on data in array X
    def fit_profile(self, X, label=''):
        try:
            assert label # Make sure it isn't empty
        except AssertionError as e:
            e.args += ('Input label is empty.')
            raise

        if self.speaker_profiles.get(label) == None:
            self.logger.info('Creating new model for class'.format(label))
            new_profile = GMM(n_components=self.n_components, \
                              covariance_type=self.covariance_type, \
                              max_iter=self.max_iter)
            new_profile.fit(X)
            self.speaker_profiles[label] = new_profile
            self.speaker_ids[label] = self.n_speakers
            self.n_speakers += 1
            self.speaker_labels.append(label)
        else:
            self.logger.info('Continuing fitting existing model for class'.format(label))
            self.speaker_profiles[label].fit(X)

    # Fit profile based on .wav files passed in (must be full path)
    def fit_profile_files(self, label='', files=[]):
        try:
            assert label
        except AssertionError as e:
            e.args += ('Input label is empty.')
            raise

        for file in files:
            try:
                assert file.endswith('.npy') # Can only use numpy files
            except AssertionError as e:
                e.args += ('File {0} is not a numpy file.'.format(file))
                raise

            if not os.path.exists(file):
                self.logger.warn('Filepath {0} does not exist. Skipping.'.format(file))
            else:
                X = np.load(file)
                self.fit_profile(X,label)

    # Fit profile based on all .wav files in a one-level directory
    def fit_profile_directory(self, label='', dir=''):
        try:
            assert label
        except AssertionError as e:
            e.args += ('Input label is empty')
            raise

        try:
            assert os.path.exists(dir)
        except AssertionError as e:
            e.args += ('Directory {0} does not exist'.format(dir))
            raise

        files = os.listdir(dir)
        if len(files) == 0:
            self.logger.warn('Found 0 files in directory {0}. Skipping.'.format(dir))
        else:
            self.logger.info('Found {0} files in directory {1}.'.format(len(files),dir))
            self.fit_profile_files(label,files)

    # Convert array y to one-hot encoding based on length D (n_speakers)
    def to_one_hot(self, y, D):
        return np.eye(D)[y]

    # Given label, dimension of one-hot vector (D), and n_data (N),
    # return one-hot array of label id of size N
    def label_to_vector(self, label, N, D):
        id = self.speaker_ids.get(label)
        try:
            assert id != None # None is different than 0!
        except AssertionError as e:
            e.args += ('Label {0} not recognized in model.'.format(label))
            raise
        y = np.full((N),id)
        return self.to_one_hot(y,D)

    # Shuffle input data arrays in sync
    def shuffle_data(self, X, y):
        n_data = y.shape[0]
        random_indices = np.random.permutation(n_data)
        return X[random_indices], y[random_indices]

    # Do n_folds cross-validation
    def cross_validate(self, X, y, n_folds=-1):
        # We need to have enough speaker profiles
        try:
            assert y.shape[1] == self.n_speakers
        except AssertionError as e:
            e.args += ('Dimension of one-hot output does not match number of speaker models. {0} =/= {1}'.format(y.shape[1],self.n_speakers))
            raise
        n_data = y.shape[0]

        # If n_folds == -1, then do leave-one-out cross-validation
        if n_folds == -1:
            n_folds = n_data
        else:
            try:
                assert n_folds > 1 and n_folds <= n_data
            except AssertionError as e:
                e.args += ('Number of folds {0} out of range'.format(n_folds))
                raise

        # Shuffle the data (may be good to not overwrite the original?)
        X, y = self.shuffle_data(X,y)

        # Calculate number of datum in each fold
        n_split = int(n_data // n_folds)

        test_accuracies = np.zeros((n_folds),dtype='float64')
        for fold in range(n_folds):
            # Reset profiles
            self.reset_profiles()

            # Slice out the test data
            X_dev = X[fold*n_split:(fold+1)*n_split]
            y_dev = y[fold*n_split:(fold+1)*n_split]

            # All other data is used for training
            X_train = np.concatenate((X[:fold*n_split],X[(fold+1)*n_split:]),axis=0)
            y_train = np.concatenate((y[:fold*n_split],y[(fold+1)*n_split:]),axis=0)

            # Train models
            self.train(X_train, y_train)

            # Test models
            test_accuracies[fold] = self.test(X_dev, y_dev)

        return test_accuracies

    # Train each model based on input y
    def train(self, X, y):
        D = y.shape[1]
        try:
            assert D == self.n_speakers
        except AssertionError as e:
            e.args += ('Dimension of one-hot output does not match number of speaker models. {0} =/= {1}'.format(D,self.n_speakers))
            raise

        for speaker_id in range(self.n_speakers):
            # Look for every data point that has one-hot at speaker_id
            # Flatten is used because it has an extra axis (1) for some reason
            data_indices = np.argwhere(y[:,speaker_id] == 1).flatten()
            # We don't need the one-hot vectors
            X_speaker = X[data_indices]
            label = self.speaker_labels[speaker_id]
            try:
                assert self.speaker_ids[label] == speaker_id
            except AssertionError as e:
                e.args += ('Speaker {0} ID mismatch. {0} =/= {1}'.format(speaker_id,self.speaker_ids[label]))
                raise

            self.speaker_profiles[label].fit(X_speaker)

    # Gets argmax of log likelihoods and compares to ground truth
    def test(self, X, y):
        N = y.shape[0]
        D = y.shape[1]
        try:
            assert D == self.n_speakers
        except AssertionError as e:
            e.args += ('Dimension of one-hot output does not match number of speaker models. {0} =/= {1}'.format(D,self.n_speakers))
            raise
        likelihoods = self.evaluate(X)
        predictions = np.eye(D)[np.argmax(likelihoods,axis=1)]
        n_correct = np.sum(predictions*y)
        accuracy = n_correct/N
        return accuracy

    # Run all models on input data and get likelihoods
    def evaluate(self, X):
        N = X.shape[0]
        predictions = np.zeros((N,self.n_speakers))
        for speaker_id in range(self.n_speakers):
            label = self.speaker_labels[speaker_id]
            try:
                assert self.speaker_ids[label] == speaker_id
            except AssertionError as e:
                e.args += ('Speaker {0} ID mismatch. {0} =/= {1}'.format(speaker_id,self.speaker_ids[label]))
                raise

            predictions[:,speaker_id] = self.speaker_profiles[label].score_samples(X)
        return predictions
