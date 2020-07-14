import numpy as np
import os
from sklearn.mixture import GaussianMixture as GMM
from scipy.io import wavfile
from librosa.feature import delta

class GMMClassifier:

    def __init__(self, data_directory=''):
        self.speech_profiles = [] # List of sklearn GMM's
        self.classes = os.listdir(data_directory)
        self.n_classes = len(self.classes)

        self.n_ccs = 13
        self.sample_rate = 16000
        self.n_components = 32

        for class_id in range(self.n_classes):
            class_directory = os.path.join(data_directory,self.classes[class_id])
            class_files = os.listdir(class_directory)
            gmm = GMM(n_components=self.n_components,covariance_type='tied')
            for filename in class_files:
                filepath = os.path.join(class_directory,filename)
                sample_rate, signal = wavfile.read(filepath)
                assert sample_rate == self.sample_rate

                mfcc_feats = mfcc(signal=signal,samplerate=sample_rate,winfunc=np.hamming)

                # Scale mfcc's (CMVN)
                mean = np.mean(mfcc_feats,axis=2)
                std = np.std(mfcc_feats,axis=2)
                mfcc_feats = (mfcc_feats - mean)/std

                delta_feats = delta(mfcc_feats,order=1)
                delta2_feats = delta(mfcc_feats,order=2)

                x_train = np.zeros((mfcc_feats.shape[0],(self.n_ccs*3)))
                x_train[:,0:self.n_ccs] = mfcc_feats
                x_train[:,self.n_ccs:2*self.n_ccs] = delta_feats
                x_train[:,2*self.n_ccs:3*self.n_ccs] = delta2_feats

                gmm.fit(x_train)

            self.speech_profiles.append(gmm)

    # Add a specific speech profile
    def add_profile(self, label='', profile_directory=''):
        pass

    def predict_profile(self, test_directory=''):
        pass


def main():

    data_directory = 'profile_data/'
    test_directory = 'test_data/'

    classifier = GMMClassifier(data_directory=data_directory)
    metrics = classifier.predict_profile(test_directory=test_directory)

    # TODO: Print metrics

if __name__ == '__main__':
    main()
