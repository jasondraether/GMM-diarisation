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
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread
from itertools import chain, combinations, product
from time import sleep

class GMMClassifier:

    # Add some way to manipulate which files to use? I'm not entirely sure how to do this the best way
    def __init__(self, data_directory='', from_directory=True, use_emphasis=True, normalize_signal=True, \
    normalize_mfcc=True, use_deltas=True, trim_silence=True, \
    pad_silence=True, n_ccs=13, n_components=32, win_len=0.025, \
    win_step=0.01, frame_length=512, frame_skip=256, top_db=30):
        self.speaker_profiles = []
        self.n_classes = 0
        self.classes = []
        self.train_corpus_dict = {}

        # For preprocessing
        self.use_emphasis = use_emphasis # Pre-emphasize filter
        self.emphasis_coefficient = 0.97
        self.normalize_signal = normalize_signal
        self.energy_multiplier = 0.05
        self.energy_range = 100

        # For MFCC's
        self.n_ccs = n_ccs
        self.sample_rate = 48000
        self.epsilon = 1e-10
        self.normalize_mfcc = normalize_mfcc
        self.use_deltas = use_deltas
        self.win_len = win_len
        self.win_step = win_step

        # For sklearn GMM
        self.n_components = n_components
        self.covariance_type = 'tied'

        # For librosa.effects.split
        self.frame_length = frame_length
        self.frame_skip = frame_skip
        self.top_db = top_db
        self.trim_silence = trim_silence
        self.pad_silence = pad_silence
        self.pad_length = 100 # 100 ms padded EACH side (200 ms total)

        # Debugging
        self.graph = False

        if os.path.exists(data_directory) and from_directory == True:
            classes = os.listdir(data_directory)
            n_classes = len(classes)
            for class_id in range(n_classes):
                training_filenames = []
                class_directory = os.path.join(data_directory,classes[class_id])
                class_files = os.listdir(class_directory)
                for class_file in class_files:
                    filename = os.path.join(class_directory,class_file)
                    training_filenames.append(filename)
                self.add_profile(label=classes[class_id],files=training_filenames)

    def add_profile(self, label='', files=''):

        self.train_corpus_dict[label] = 0

        assert len(files) > 0

        gmm = GMM(n_components=self.n_components,covariance_type=self.covariance_type)

        for filepath in files:

            #print(filepath)
            assert os.path.exists(filepath)

            sample_rate, signal = wavfile.read(filepath)
            assert sample_rate == self.sample_rate

            signal = self.preprocess_signal(signal)

            x_train = self.calculate_features(signal=signal,sample_rate=sample_rate)

            gmm.fit(x_train)
            self.train_corpus_dict[label] += x_train.shape[0]

            #print("BIC for class {0}: {1}".format(label,gmm.bic(x_train)))

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

    def clear_silence(self, signal):

        indices = split(y=signal,top_db=self.top_db, \
                        frame_length=self.frame_length,hop_length=self.frame_skip)

        signal_slices = []
        for index in indices:
            signal_slice = signal[index[0]:index[1]]

            if self.pad_silence:
                num_samples = int((self.sample_rate*self.pad_length)/1000.0)
                signal_length = signal_slice.shape[0]
                padded_signal = np.zeros((signal_length+(2*num_samples)))
                padded_signal[num_samples:signal_length+num_samples] = signal_slice
                signal_slice = padded_signal

            signal_slices.append(signal_slice)

        return np.concatenate(signal_slices,axis=0)

    def calculate_features(self, signal, sample_rate):

        # We do our own preemphasis filter
        # TODO: See if librosa's results end up being different than python_speech_features
        mfcc_feats = mfcc(signal,samplerate=sample_rate,winfunc=np.hamming,numcep=self.n_ccs,nfft=4096,preemph=0,winlen=self.win_len,winstep=self.win_step)

        if self.normalize_mfcc:
            mean = np.mean(mfcc_feats,axis=0)
            std = np.std(mfcc_feats,axis=0)
            # mfcc_feats = (mfcc_feats - mean)/(std + self.epsilon) # This is the correct way to normalize but has weird results
            mfcc_feats = mfcc_feats - mean

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

    def preprocess_signal(self, signal):

        if self.graph:
            self.plot_signal(signal,sample_rate,'Starting Test Signal')

        if self.use_emphasis:
            signal = np.append(signal[0], signal[1:] - self.emphasis_coefficient * signal[:-1])
            if self.graph:
                self.plot_signal(signal,sample_rate,'Emphasized Test Signal')

        if self.trim_silence:
            signal = self.clear_silence(signal)
            if self.graph:
                self.plot_signal(signal,sample_rate,'Silenced Test Signal')

        if self.normalize_signal:
            signal = self.energy_normalization(signal)
            if self.graph:
                self.plot_signal(signal,sample_rate,'Normalized Test Signal')

        if self.graph:
            self.plot_signal(signal,sample_rate,'Final Test Signal')

        return signal

    def predict_from_file(self, filepath=''):

        assert os.path.exists(filepath)

        sample_rate, signal = wavfile.read(filepath)

        return self.predict_from_array(signal=signal,sample_rate=sample_rate)

    def predict_from_array(self, signal, sample_rate):

        assert sample_rate == self.sample_rate

        signal = self.preprocess_signal(signal)

        x_test = self.calculate_features(signal=signal,sample_rate=sample_rate)

        log_likelihood = np.zeros((x_test.shape[0],self.n_classes))

        for class_id in range(self.n_classes):
            gmm = self.speaker_profiles[class_id]
            log_likelihood[:,class_id] = gmm.score_samples(x_test)

        return log_likelihood

    def get_classes(self):
        return self.classes

    def get_train_corpus_dict(self):
        return self.train_corpus_dict

    def get_sample_rate(self):
        return self.sample_rate


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1)))

def main():

    data_directory = 'profile_data/'
    test_directory = 'test_data/'

    tune_hyperparameters = False
    from_directory = False
    tune_files = True

    interval_length = 0.5 # Half second intervals

    # Best parameters so far
    use_emphasis=True
    normalize_signal=True
    normalize_mfcc=False  # False
    use_deltas=True
    trim_silence=False # False
    pad_silence = False # False
    n_ccs = 13
    n_components = 4
    win_len = 0.025
    win_step = 0.01
    frame_length = 512
    frame_skip = 256
    top_db = 30

    # File tuning
    # if os.path.exists(data_directory):
    #     classes = os.listdir(data_directory)
    #     n_classes = len(classes)
    #     for class_id in range(n_classes):
    #         class_directory = os.path.join(data_directory,classes[class_id])
    #         self.add_profile(label=classes[class_id],profile_directory=class_directory)


    # Hyperparameter tuning

    if tune_hyperparameters:
        best_accuracy = 0
        best_parameters = []

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
                            for f in range(2):
                                if f == 0:
                                    n_ccs = 13
                                elif f == 1:
                                    n_ccs = 20
                                for g in range(1,33):
                                    n_components = g
                                    for h in range(2,12):
                                        win_len = 0.005*h
                                        for i in range(1,12):
                                            win_step = 0.0025*i
                                            for j in range(7,13):
                                                frame_length = 2**j
                                                for k in range(5,11):
                                                    frame_skip = 2**k
                                                    for l in range(20, 45, 5):
                                                        top_db = l

                                                        # use os.rename
                                                        # Start
                                                        classifier = GMMClassifier(data_directory=data_directory, from_directory=from_directory, use_emphasis=use_emphasis, \
                                                                                   normalize_signal=normalize_signal, normalize_mfcc=normalize_mfcc, \
                                                                                   use_deltas=use_deltas, trim_silence=trim_silence, pad_silence=pad_silence, \
                                                                                   n_ccs=n_ccs, n_components=n_components, win_len=win_len, \
                                                                                   win_step=win_step, frame_length=frame_length, frame_skip=frame_skip, \
                                                                                   top_db=top_db)

                                                        classes = classifier.get_classes()
                                                        train_corpus_dict = classifier.get_train_corpus_dict()
                                                        sample_rate = classifier.get_sample_rate()
                                                        sample_length = int(interval_length * sample_rate)

                                                        tests = []
                                                        test_corpus_dict = {}

                                                        assert os.path.exists(test_directory)
                                                        test_classes = os.listdir(test_directory)
                                                        for test_class in test_classes:
                                                            assert test_class in classes
                                                            test_corpus_dict[test_class] = 0
                                                            test_class_directory = os.path.join(test_directory,test_class)
                                                            test_class_files = os.listdir(test_class_directory)
                                                            for test_file in test_class_files:
                                                                test_filename = os.path.join(test_class_directory,test_file)
                                                                tests.append([test_filename,test_class])

                                                        n_correct = 0
                                                        n_total = 0

                                                        for test in tests:
                                                            log_likelihood = classifier.predict_from_file(test[0])

                                                            n_data = log_likelihood.shape[0]
                                                            assert n_data > 0

                                                            for llh in log_likelihood:
                                                                prediction = np.argmax(llh)
                                                                if classes[prediction] == test[1]:
                                                                    n_correct += 1

                                                            test_corpus_dict[test[1]] += n_data
                                                            n_total += n_data

                                                        accuracy = n_correct / n_total
                                                        parameters = [accuracy,use_emphasis,normalize_signal,normalize_mfcc,use_deltas,trim_silence,n_ccs,n_components,win_len,win_step,frame_length,frame_skip,top_db]
                                                        print(parameters, end='\r')

                                                        if accuracy > best_accuracy:
                                                            best_accuracy = accuracy
                                                            best_parameters = parameters
                                                            print("\nNew best accuracy parameters:", best_parameters)

                                                        if trim_silence == False:
                                                            break
                                                    if trim_silence == False:
                                                        break
                                                if trim_silence == False:
                                                    break

                                                        # End
    elif tune_files:


        file_powersets = []
        if os.path.exists(data_directory):
            classes = os.listdir(data_directory)
            n_classes = len(classes)
            for class_id in range(n_classes):
                training_filenames = []
                class_directory = os.path.join(data_directory,classes[class_id])
                class_files = os.listdir(class_directory)
                for class_file in class_files:
                    filename = os.path.join(class_directory,class_file)
                    training_filenames.append([filename,class_id])

                training_powerset = powerset(training_filenames)
                #training_powerset = list(filter(None,training_powerset))
                temp_powerset = []
                for t in training_powerset:
                    inner_set = []
                    for x in t:
                        inner_set.append(x)
                    temp_powerset.append(inner_set)

                training_powerset = temp_powerset

                print('=====')
                for x in training_powerset:
                    print(x)
                print('=====')

                file_powersets.append(training_powerset)

        file_product = list(product(*file_powersets))

        temp_product = []
        for t in file_product:
            inner_product = []
            for x in t:
                inner_product.append(x)
            temp_product.append(inner_product)
        file_product = temp_product

        for files_list in file_product:

            viable = True

            classifier = GMMClassifier(data_directory=data_directory, from_directory=from_directory, use_emphasis=use_emphasis, \
                                       normalize_signal=normalize_signal, normalize_mfcc=normalize_mfcc, \
                                       use_deltas=use_deltas, trim_silence=trim_silence, pad_silence=pad_silence, \
                                       n_ccs=n_ccs, n_components=n_components, win_len=win_len, \
                                       win_step=win_step, frame_length=frame_length, frame_skip=frame_skip, \
                                       top_db=top_db)

            saved_filenames = []
            for class_id in range(n_classes):
                training_filenames = []
                for sublist in files_list:
                    for f in sublist:
                        if f[1] == class_id:
                            training_filenames.append(f[0])

                saved_filenames.append(training_filenames)
                if len(training_filenames) == 0:
                    viable = False
                else:
                    classifier.add_profile(label=classes[class_id],files=training_filenames)


            if viable:
                print("Evaluating")
                classes = classifier.get_classes()
                train_corpus_dict = classifier.get_train_corpus_dict()
                sample_rate = classifier.get_sample_rate()
                sample_length = int(interval_length * sample_rate)

                tests = []
                test_corpus_dict = {}

                assert os.path.exists(test_directory)
                test_classes = os.listdir(test_directory)
                for test_class in test_classes:
                    assert test_class in classes
                    test_corpus_dict[test_class] = 0
                    test_class_directory = os.path.join(test_directory,test_class)
                    test_class_files = os.listdir(test_class_directory)
                    for test_file in test_class_files:
                        test_filename = os.path.join(test_class_directory,test_file)
                        tests.append([test_filename,test_class])

                n_correct = 0
                n_total = 0

                for test in tests:
                    test_correct = 0
                    log_likelihood = classifier.predict_from_file(test[0])

                    n_data = log_likelihood.shape[0]
                    assert n_data > 0

                    for llh in log_likelihood:
                        prediction = np.argmax(llh)
                        if classes[prediction] == test[1]:
                            test_correct += 1

                    test_corpus_dict[test[1]] += n_data
                    n_correct += test_correct
                    n_total += n_data
                    #print("File {0} of class {1} had accuracy {2}".format(test[0],test[1],test_correct/n_data))

                accuracy = n_correct / n_total

                print('Total Accuracy: {0}'.format(accuracy))
                print('Filenames: {0}\n'.format(saved_filenames))

    # # Perform on actual unknown file
    # podcast_filename = 'unknown_data/196.wav'
    # animation_directory = 'animation_data/'
    # video_filename = 'animation.mp4'
    # width = 1920
    # height = 1080
    # FPS = 60
    # max_seconds = 100
    #
    # still_filename = 'still.png'
    # still_frame = imread(os.path.join(animation_directory,still_filename))
    #
    # frames = []
    #
    # for class_id in range(len(classes)):
    #     class_frame_filename = os.path.join(animation_directory,(classes[class_id]+'.png'))
    #     frame = imread(class_frame_filename)
    #     frames.append(frame)
    #
    # fourcc = VideoWriter_fourcc(*'mp4v')
    # video = VideoWriter(video_filename, fourcc, float(FPS), (width,height))
    #
    # sample_rate, podcast = wavfile.read(podcast_filename)
    # n_samples = podcast.shape[0]
    # seconds = n_samples//sample_rate
    # frames_per_interval = int(interval_length*FPS)
    #
    # timestamp_labels = []
    #
    # for i in range(0,n_samples-sample_length,sample_length):
    #     log_likelihood = classifier.predict_from_array(signal=podcast[i:i+sample_length],sample_rate=sample_rate)
    #
    #     prediction = np.argmax(np.mean(log_likelihood,axis=0))
    #
    #     animation_range_lower = frames_per_interval//3
    #     animation_range_upper = (frames_per_interval*2)//3
    #     for f in range(frames_per_interval):
    #         if f > animation_range_lower and f < animation_range_upper:
    #             video.write(still_frame)
    #         else:
    #             video.write(frames[prediction])
    #
    #
    #     current_time = i/sample_rate
    #
    #     print("Wrote {0} frames for time {1}".format(classes[prediction],i/sample_rate))
    #
    #     if current_time >= max_seconds:
    #         break
    #
    #     # starting_timestamp = i/sample_rate
    #     # ending_timestamp = (i+sample_length)/sample_rate
    #     #
    #     # timestamp_labels.append([prediction, starting_timestamp, ending_timestamp])
    #
    # video.release()


if __name__ == '__main__':
    main()
