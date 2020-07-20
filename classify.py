import numpy as np
import os
from sklearn.mixture import GaussianMixture as GMM
from scipy.io import wavfile
from librosa.feature import delta
from python_speech_features import mfcc
from librosa.effects import split
from pydub.playback import play
from pydub import AudioSegment
import time
import matplotlib.pyplot as plt
from matplotlib import cm



class GMMClassifier:

    def __init__(self, data_directory='', use_emphasis=True, normalize_signal=True, normalize_mfcc=True, use_deltas=True, trim_silence=True):
        self.speaker_profiles = []
        self.n_classes = 0
        self.classes = []
        self.corpus_dict = {}

        # For preprocessing
        self.use_emphasis = use_emphasis # Pre-emphasize filter
        self.emphasis_coefficient = 0.97
        self.normalize_signal = normalize_signal
        self.energy_multiplier = 0.05
        self.energy_range = 100

        # For MFCC's
        self.n_ccs = 13
        self.sample_rate = 48000
        self.epsilon = 1e-10
        self.normalize_mfcc = normalize_mfcc
        self.use_deltas = use_deltas

        # For sklearn GMM
        self.n_components = 32
        self.covariance_type = 'tied'

        # For librosa.effects.split
        self.frame_length = 2048
        self.frame_skip = 512
        self.top_db = 30
        self.trim_silence = trim_silence

        if os.path.exists(data_directory):
            classes = os.listdir(data_directory)
            n_classes = len(classes)
            for class_id in range(n_classes):
                class_directory = os.path.join(data_directory,classes[class_id])
                self.add_profile(label=classes[class_id],profile_directory=class_directory)

    def add_profile(self, label='', profile_directory=''):

        self.corpus_dict[label] = 0

        assert os.path.exists(profile_directory)
        files = os.listdir(profile_directory)

        gmm = GMM(n_components=self.n_components,covariance_type=self.covariance_type)

        for filename in files:

            filepath = os.path.join(profile_directory,filename)
            assert os.path.exists(filepath)

            sample_rate, signal = wavfile.read(filepath)
            assert sample_rate == self.sample_rate

            # self.plot_signal(signal,sample_rate)

            if self.use_emphasis:
                signal = np.append(signal[0], signal[1:] - self.emphasis_coefficient * signal[:-1])
                self.plot_signal(signal,sample_rate,'Emphasized Signal')

            if self.trim_silence:
                indices = split(y=signal,top_db=self.top_db, \
                                frame_length=self.frame_length,hop_length=self.frame_skip)

                x_train = []
                for index in indices:
                    signal_slice = signal[index[0]:index[1]] # TODO: Add zero padding so less awkward
                    self.plot_signal(signal_slice,sample_rate,'Silenced Signal')
                    if self.normalize_signal:
                        signal_slice = self.energy_normalization(signal_slice)
                        self.plot_signal(signal_slice,sample_rate,'Silenced Normalized Signal')

                    x_train_slice = self.calculate_features(signal=signal_slice,sample_rate=sample_rate)
                    x_train.append(x_train_slice)

                x_train = np.concatenate(x_train,axis=0)
            else:
                if self.normalize_signal:
                    signal = self.energy_normalization(signal)
                    self.plot_signal(signal,sample_rate,'Normalized Signal')

                x_train = self.calculate_features(signal=signal,sample_rate=sample_rate)

            gmm.fit(x_train)
            self.corpus_dict[label] += x_train.shape[0]

        self.speaker_profiles.append(gmm)
        self.classes.append(label)
        self.n_classes += 1

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

    def plot_signal(self, signal, sample_rate, title='Signal'):
        Time=np.linspace(0, len(signal)/sample_rate, num=len(signal))
        plt.figure(figsize=(20,5))
        plt.plot(Time,signal)
        plt.title(title)
        plt.ylabel('Amplitude')
        plt.xlabel('Time (Sec)')
        plt.show()

    def energy_normalization(self, signal):

        energy = [a**2 for a in signal]
        voiced_threshold = self.energy_multiplier*np.mean(energy)
        clean_samples = [0]

        for sample_set in range(0, len(signal)-self.energy_range, self.energy_range):
            sample_set_th = np.mean(energy[sample_set:sample_set+self.energy_range])
            if sample_set_th > voiced_threshold:
                clean_samples.extend(signal[sample_set:sample_set+self.energy_range])

        return np.array(clean_samples)

    def calculate_features(self, signal, sample_rate):

        mfcc_feats = mfcc(signal,samplerate=sample_rate,winfunc=np.hamming,numcep=self.n_ccs,nfft=2048)

        if self.normalize_mfcc:
            mean = np.mean(mfcc_feats,axis=0)
            std = np.std(mfcc_feats,axis=0)
            mfcc_feats = (mfcc_feats - mean)/(std + self.epsilon)

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


    def predict_from_file(self, filepath=''):

        likelihoods = []

        assert os.path.exists(filepath)

        sample_rate, signal = wavfile.read(filepath)
        assert sample_rate == self.sample_rate

        if self.use_emphasis:
            signal = np.append(signal[0], signal[1:] - self.emphasis_coefficient * signal[:-1])

        if self.trim_silence:
            indices = split(y=signal,top_db=self.top_db, \
                            frame_length=self.frame_length,hop_length=self.frame_skip)

            for index in indices:
                signal_slice = signal[index[0]:index[1]] # TODO: Add zero padding so less awkward

                if self.normalize_signal:
                    signal_slice = self.energy_normalization(signal_slice)

                x_test = self.calculate_features(signal=signal_slice,sample_rate=sample_rate)

                n_test_data = x_test.shape[0]
                log_likelihood = np.zeros((n_test_data,self.n_classes))

                for class_id in range(self.n_classes):
                    gmm = self.speaker_profiles[class_id]
                    log_likelihood[:,class_id] = gmm.score_samples(x_test)

                likelihoods.append(log_likelihood)

        else:
            if self.normalize_signal:
                signal = self.energy_normalization(signal)

            x_test = self.calculate_features(signal=signal,sample_rate=sample_rate)

            n_test_data = x_test.shape[0]
            log_likelihood = np.zeros((n_test_data,self.n_classes))

            for class_id in range(self.n_classes):
                gmm = self.speaker_profiles[class_id]
                log_likelihood[:,class_id] = gmm.score_samples(x_test)

            likelihoods.append(log_likelihood)

        return likelihoods

    def predict_from_array(self, signal, sample_rate):
        pass



    def get_classes(self):
        return self.classes

    def get_corpus_dict(self):
        return self.corpus_dict

    def get_sample_rate(self):
        return self.sample_rate


def main():

    data_directory = 'profile_data/'
    test_directory = 'test_data/'

    interval_length = 0.5 # Half second intervals

    accuracies = []

    use_emphasis=True
    normalize_signal=True
    normalize_mfcc=False
    use_deltas=True
    trim_silence=False

    for a in range(2):
        use_emphasis = bool(a)
        for b in range(2):
            normalize_signal = bool(b)
            for c in range(2):
                normalize_mfcc = bool(c)
                for d in range(2):
                    use_deltas = bool(d)
                    for e in range(2):
                        trim_silence = bool(e)

                        classifier = GMMClassifier(data_directory=data_directory,use_emphasis=use_emphasis, \
                                                   normalize_signal=normalize_signal, normalize_mfcc=normalize_mfcc, \
                                                   use_deltas=use_deltas, trim_silence=trim_silence)

                        classes = classifier.get_classes()
                        corpus_dict = classifier.get_corpus_dict()
                        sample_rate = classifier.get_sample_rate()

                        tests = []

                        assert os.path.exists(test_directory)
                        test_classes = os.listdir(test_directory)
                        for test_class in test_classes:
                            assert test_class in classes
                            test_class_directory = os.path.join(test_directory,test_class)
                            test_class_files = os.listdir(test_class_directory)
                            for test_file in test_class_files:
                                test_filename = os.path.join(test_class_directory,test_file)
                                tests.append([test_filename,test_class])

                        for test in tests:
                            n_correct = n_total = 0.0
                            likelihoods = classifier.predict_from_file(filepath=test[0])
                            for l in likelihoods:
                                n_data = l.shape[0]
                                for n in range(n_data):
                                    prediction = np.argmax(l[n])
                                    n_total += 1.0

                                    if classes[prediction] == test[1]:
                                        n_correct += 1.0

                            if n_total == 0.0:
                                print("Error: n_total == 0! No tests run for file {0}".format(test[0]))
                            else:
                                accuracy = n_correct/n_total
                                print("File {0} of class {1} had per-sample accuracy {2}".format(test[0],test[1],accuracy))
                                accuracies.append([test[0],test[1],accuracy,use_emphasis,normalize_signal,normalize_mfcc,use_deltas,trim_silence])



                        print("Corpus dictionary:{0}".format(corpus_dict))

    for accuracy in accuracies:
        print(accuracy)


if __name__ == '__main__':
    main()

#
# fname = 'test.wav'
#
# sample_rate, signal = wavfile.read(fname)
#
# active_indices = split(y=signal,top_db=30)
# i = 0
# for index in active_indices:
#     write_dest = 'out'+str(i)+'.wav'
#     cut_signal = signal[index[0]:index[1]]
#
#     Time=np.linspace(0, len(cut_signal)/sample_rate, num=len(cut_signal))
#
#     plt.figure(figsize=(20,5))
#     plt.plot(Time,cut_signal)
#     plt.title('Emphasized Signal without Silence (Librosa)')
#     plt.ylabel('Amplitude')
#     plt.xlabel('Time (Sec)')
#     plt.show()
#
#     time.sleep(1)
    #
    # wavfile.write(write_dest,sample_rate,cut_signal)
    # audio_segment = AudioSegment.from_wav(write_dest)
    # play(audio_segment)
    # i+=1
    # time.sleep(1)


# audio_segment = AudioSegment(
#     signal.tobytes(),
#     frame_rate=sample_rate,
#     sample_width=signal.dtype.itemsize,
#     channels=1
# )
