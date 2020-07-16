import numpy as np
import os
from sklearn.mixture import GaussianMixture as GMM
from scipy.io import wavfile
from librosa.feature import delta
from python_speech_features import mfcc
from pydub.silence import split_on_silence
from pydub import AudioSegment
from pydub.playback import play


class GMMClassifier:

    def __init__(self, data_directory=''):
        self.speech_profiles = [] # List of sklearn GMM's
        self.n_classes = 0
        self.classes = []
        self.corpus_size = {}

        self.n_ccs = 13
        self.sample_rate = 48000
        self.n_components = 32
        self.covariance_type = 'tied'
        self.epsilon = 1e-10

        if os.path.exists(data_directory):
            print("Initializing speech profiles using directory {0}".format(data_directory))
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

        self.corpus_size[label] = 0

        assert os.path.exists(profile_directory)
        files = os.listdir(profile_directory)
        gmm = GMM(n_components=self.n_components,covariance_type=self.covariance_type)

        for filename in files:
            print("Fitting profile {0} with audio file {1}".format(label,filename))

            filepath = os.path.join(profile_directory,filename)
            assert os.path.exists(filepath)

            sample_rate, signal = wavfile.read(filepath)
            pydub_signal = AudioSegment.from_wav(filepath)
            assert sample_rate == self.sample_rate
            # signal = signal.astype(float)

            split_signal = split_on_silence(audio_segment=pydub_signal,min_silence_len=1000,silence_thresh=-40,keep_silence=100,seek_step=1) # More parameters to this, see: https://github.com/jiaaro/pydub/blob/master/pydub/silence.py

            for signal_slice in split_signal:
                # play(signal_slice)
                samples = signal_slice.get_array_of_samples()
                signal = np.frombuffer(samples,dtype=np.int16)
                #signal = np.array(samples)

                x_train = self.calculate_features(signal=signal,sample_rate=sample_rate)

                # Resumes where left off, fix this so we don't just throw out samples...
                if x_train.shape[0] > 1:
                    print("{0} samples in this signal slice fitting session".format(x_train.shape[0]))
                    print(x_train)
                    gmm.fit(x_train)
                    self.corpus_size[label] += x_train.shape[0]

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
            mfcc_feats = (mfcc_feats - mean)/(std+self.epsilon)

        delta_feats = delta(mfcc_feats,order=1)
        delta2_feats = delta(mfcc_feats,order=2)

        features = np.zeros((mfcc_feats.shape[0],(self.n_ccs*3)))
        features[:,0:self.n_ccs] = mfcc_feats
        features[:,self.n_ccs:2*self.n_ccs] = delta_feats
        features[:,2*self.n_ccs:3*self.n_ccs] = delta2_feats

        return features

    def predict_profile(self, filepath=''):

        # Score samples for log likelihood (i.e., how plausible is this model?)
        # Predict proba for cluster probability (will sum to 1 for each model)

        assert os.path.exists(filepath)
        pydub_signal = AudioSegment.from_wav(filepath)
        split_signal = split_on_silence(audio_segment=pydub_signal,min_silence_len=1000,silence_thresh=-40,keep_silence=100,seek_step=1) # More parameters to this, see: https://github.com/jiaaro/pydub/blob/master/pydub/silence.py

        slice_id = 0
        likelihoods = []

        for signal_slice in split_signal:
            sample_rate = signal_slice.frame_rate
            assert sample_rate == self.sample_rate

            samples = signal_slice.get_array_of_samples()
            signal = np.frombuffer(samples,dtype=np.int16)

            x_test = self.calculate_features(signal=signal,sample_rate=sample_rate,normalize=True)
            log_likelihood = np.zeros((x_test.shape[0],self.n_classes))

            print(self.n_classes)
            for class_id in range(self.n_classes):
                gmm = self.speech_profiles[class_id]
                log_likelihood[:,class_id] = gmm.score_samples(x_test)

            likelihoods.append(log_likelihood)
            average_log_likelihood = np.mean(log_likelihood,axis=0)

            class_prediction = np.argmax(average_log_likelihood)

            print("File {0} of slice {1} predicted to be {2}".format(filepath,slice_id,self.classes[class_prediction]))

        return likelihoods

    def get_classes(self):
        return self.classes

    def get_corpus_size(self):
        return self.corpus_size

def main():

    data_directory = 'profile_data/'
    test_directory = 'test_data/'

    # First is matt, second is ryan
    # test_files = [['test_data/test1.wav','matt'],['test_data/test2.wav','ryan']]

    classifier = GMMClassifier(data_directory=data_directory)

    classes = classifier.get_classes()
    corpus_size = classifier.get_corpus_size()

    print("Corpus dictionary:{0}".format(corpus_size))

    test_files = []
    # Create test corpus (might be easier to remake structure so we just have data->classes->test/train)
    assert os.path.exists(test_directory)
    test_classes = os.listdir(test_directory)
    for test_class in test_classes:
        assert test_class in classes
        class_directory = os.path.join(test_directory,test_class)
        class_files = os.listdir(class_directory)
        for file in class_files:
            filename = os.path.join(class_directory,file)
            test_files.append([filename,test_class])

    # Check accuracy on corpus
    for test in test_files:
        file_correct = 0.0
        file_total = 0.0
        likelihoods = classifier.predict_profile(filepath=test[0])
        for log_likelihood in likelihoods:
            for i in range(log_likelihood.shape[0]):
                pred = np.argmax(log_likelihood[i])
                file_total += 1.0

                print(classes[pred],test[1],log_likelihood[i])
                if classes[pred] == test[1]:
                    file_correct += 1.0

        print("File {0} of class {1} had per-sample accuracy {2}".format(test[0],test[1],file_correct/file_total))

    # TODO: Print metrics (BIC, )

if __name__ == '__main__':
    main()
