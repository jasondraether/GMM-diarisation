import numpy as np
import os
from sklearn.mixture import GaussianMixture as GMM
from scipy.io import wavfile
from librosa.feature import delta
from python_speech_features import mfcc


class GMMClassifier:

    def __init__(self, data_directory=''):
        self.speech_profiles = [] # List of sklearn GMM's
        self.n_classes = 0
        self.classes = []

        self.n_ccs = 13
        self.sample_rate = 48000
        self.n_components = 32

        if os.path.exists(data_directory):
            print("Initializing model using directory {0}".format(data_directory))
            classes = os.listdir(data_directory)
            n_classes = len(classes)
            for class_id in range(n_classes):
                class_directory = os.path.join(data_directory,classes[class_id])
                self.add_profile(label=classes[class_id],profile_directory=class_directory)
        else:
            print("Initializing empty model")

    # Add a specific speech profile for directory
    def add_profile(self, label='', profile_directory=''):

        print("Adding speech profile for {0} in directory {1}".format(label,profile_directory))

        assert os.path.exists(profile_directory)
        files = os.listdir(profile_directory)
        gmm = GMM(n_components=self.n_components,covariance_type='tied')

        for filename in files:
            print("Fitting profile {0} with audio file {1}".format(label,filename))

            filepath = os.path.join(profile_directory,filename)
            assert os.path.exists(filepath)

            sample_rate, signal = wavfile.read(filepath)
            assert sample_rate == self.sample_rate

            x_train = self.calculate_features(signal=signal,sample_rate=sample_rate)

            # Resumes where left off
            gmm.fit(x_train)

            # TODO: Use labels for clusters (laughing, yelling, normal, etc.)
            # TODO: Add BIC to maximize training v samples

        self.speech_profiles.append(gmm)
        self.classes.append(label)
        self.n_classes += 1

    def calculate_features(self,signal,sample_rate,normalize=True):

        mfcc_feats = mfcc(signal=signal,samplerate=sample_rate,winfunc=np.hamming,numcep=self.n_ccs,nfft=2048)

        # Scale mfcc's (CMVN)
        if normalize:
            mean = np.mean(mfcc_feats,axis=0)
            std = np.std(mfcc_feats,axis=0)
            mfcc_feats = (mfcc_feats - mean)/std

        delta_feats = delta(mfcc_feats,order=1)
        delta2_feats = delta(mfcc_feats,order=2)

        features = np.zeros((mfcc_feats.shape[0],(self.n_ccs*3)))
        features[:,0:self.n_ccs] = mfcc_feats
        features[:,self.n_ccs:2*self.n_ccs] = delta_feats
        features[:,2*self.n_ccs:3*self.n_ccs] = delta2_feats

        return features

    def predict_profile(self, filename='', time_slice=0):

        # Score samples for log likelihood (i.e., how plausible is this model?)
        # Predict proba for cluster probability (will sum to 1 for each model)

        assert os.path.exists(filename)

        sample_rate, signal = wavfile.read(filename)
        assert sample_rate == self.sample_rate

        x_test = self.calculate_features(signal=signal,sample_rate=sample_rate,normalize=True)
        log_likelihood = np.zeros((x_test.shape[0],self.n_classes))

        for class_id in range(self.n_classes):
            gmm = self.speech_profiles[class_id]
            log_likelihood[:,class_id] = gmm.score_samples(x_test)

        # for i in range(x_test.shape[0]):
        #     c = np.argmax(log_likelihood[i,:])
        #     print(self.classes[c])


        average_log_likelihood = np.mean(log_likelihood,axis=0)

        class_prediction = np.argmax(average_log_likelihood)

        print("File {0} predicted to be {1}".format(filename,self.classes[class_prediction]))

        return log_likelihood, average_log_likelihood

    def get_classes(self):
        return self.classes

def main():

    data_directory = 'profile_data/'

    # First is matt, second is ryan
    test_files = [['test_data/test1.wav','matt'],['test_data/test2.wav','ryan']]

    classifier = GMMClassifier(data_directory=data_directory)
    classes = classifier.get_classes()
    for filename in test_files:
        file_correct = 0.0
        file_total = 0.0
        log_likelihood, average_log_likelihood = classifier.predict_profile(filename=filename[0])
        for i in range(log_likelihood.shape[0]):
            pred = np.argmax(log_likelihood[i])
            file_total += 1.0

            if classes[pred] == filename[1]:
                file_correct += 1.0

        print("File {0} of class {1} had per-sample accuracy {2}".format(filename[0],filename[1],file_correct/file_total))

    # TODO: Print metrics (BIC, )

if __name__ == '__main__':
    main()
