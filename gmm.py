from sklearn.mixture import GaussianMixture as GMM
import os
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc
from sklearn import preprocessing
import time
from pydub import AudioSegment
from pydub.playback import play
#from librosa.feature import mfcc
from librosa.feature import delta

n_components = 32 # Matt, Ryan, Both, Silence, Music
covariance_type = 'full'
max_iter = 10000
n_init = 3
classes = ['matt','ryan']
data_path = 'data/'
n_classes = len(classes)

num_cep = 20
win_step = 0.02
win_len = 0.05

models = []

for i in range(n_classes):
    print("Training new GMM.")
    gmm = GMM(n_components=n_components,covariance_type=covariance_type,max_iter=max_iter,n_init=n_init)
    filepath = data_path + classes[i] + '.wav'
    sample_rate, signal = wavfile.read(filepath)

    print("Data path found:", filepath)
    print("Signal of length:",len(signal),"at rate:",sample_rate)
    # print("Segmenting signal...")


    # segment_duration = 0.6
    # signal_duration = signal.shape[0]//sample_rate
    # signal_duration = round((signal_duration//segment_duration)*segment_duration,1)
    # num_arrays = int((signal_duration/segment_duration))
    # truncated_signal = signal[:int(signal_duration*sample_rate)]
    # signal_segments = np.asarray(np.split(truncated_signal,num_arrays))

    # print("Overlapping signal...")

    # overlap_duration = 0.2
    # overlap_length = int(0.2*sample_rate)
    # signal_overlap = np.zeros((num_arrays-2, sample_rate))
    # for j in range(1,num_arrays-1):
    #     signal_overlap[j-1] = np.concatenate((signal_segments[j-1,-overlap_length:],signal_segments[j],signal_segments[j+1,:overlap_length]))

    # print("Calculating scaled MFCC's...")
    # num_samples = signal_overlap.shape[0]
    # x_train = np.zeros((num_samples,input_shape[0]*input_shape[1]))
    # for datum in range(num_samples):
    #     x_train[datum] = np.hstack(preprocessing.scale(mfcc(signal=signal_overlap[datum],samplerate=sample_rate,winstep=win_step,winfunc=np.hamming)))
    #
    # print(x_train)
    # print("Fitting GMM with",num_samples,"samples...")

    num_samples = int((len(signal) // sample_rate) / win_step)

    mfcc_feats = np.zeros((num_samples,num_cep))
    delta_feats = np.zeros((num_samples,num_cep))
    delta2_feats = np.zeros((num_samples,num_cep))

    mfcc_feats = mfcc(signal=signal,numcep=num_cep,samplerate=sample_rate,winstep=win_step,winfunc=np.hamming)
    delta_feats = delta(data=mfcc_feats,order=1)
    delta2_feats = delta(data=mfcc_feats,order=2)

    x_train = np.zeros((mfcc_feats.shape[0],(num_cep*3)))

    x_train[:,0:num_cep] = mfcc_feats
    x_train[:,num_cep:2*num_cep] = delta_feats
    x_train[:,2*num_cep:3*num_cep] = delta2_feats

    x_train = np.array(preprocessing.scale(x_train))
    print("Training on:",x_train.shape[0],"samples.")

    gmm.fit(x_train)

    print("Done fitting GMM. Saving into models array.")

    models.append(gmm)

full_test_path = 'test_data/test.wav'
sample_rate, signal = wavfile.read(full_test_path)
increment = sample_rate
start_index = 0
end_index = increment
timestamp = 0.0
time_delta = increment/sample_rate

while end_index < len(signal):
    slice = signal[start_index:end_index]
    predictions = np.zeros(len(classes))

    mfcc_feat = mfcc(signal=slice,numcep=num_cep,samplerate=sample_rate,winstep=win_step,winfunc=np.hamming)
    delta_feat = delta(data=mfcc_feat,order=1)
    delta2_feat = delta(data=mfcc_feat,order=2)

    x_test = np.zeros((mfcc_feat.shape[0],num_cep*3))
    x_test[:,0:num_cep] = mfcc_feat
    x_test[:,num_cep:2*num_cep] = delta_feat
    x_test[:,2*num_cep:3*num_cep] = delta2_feat

    x_test = np.array(preprocessing.scale(x_test))

    for i in range(len(models)):
        gmm = models[i]
        predictions[i] = gmm.score(x_test)
    print("\nTime:",timestamp,"Speaker:",classes[np.argmax(predictions)],'\n')
    sound = AudioSegment(data=slice,sample_width=2,frame_rate=sample_rate,channels=1)
    play(sound)
    timestamp += time_delta
    start_index = end_index
    end_index += increment
    #time.sleep(0.5)
