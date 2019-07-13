from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, Activation
from keras import regularizers, optimizers
from keras.utils import to_categorical
import processing # My own file processing.py
import os
import numpy as np

'''
Neural network code for
audio-learning
'''
class NeuralNetwork:

    def __init__(self, input_shape=(0, 0), output_labels=None, learning_rate=0.001, decay=0.001, model_path='models/nn.model', model=None):
        self.input_shape = input_shape # Shape of spectrogram is (129, 196)
        self.output_labels = output_labels # Classes
        self.learning_rate = learning_rate # Unused
        self.decay = decay # Unused
        self.model = model # Compiled model
        self.model_path = model_path # Path to save model

    def start_tf_session():

        # Configure TF session (allow memory growth, helps with OOM error)
    	config = tf.ConfigProto()
    	config.gpu_options.allow_growth = True
    	config.gpu_options.per_process_gpu_memory_fraction = 0.4
    	self.session = tf.Session(config=config)

    def create_model(self):

        # Model instantiation
        model = Sequential()

        # First layer
        model.add(Conv1D(64, 3, input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2))

        # Second layer
        model.add(Conv1D(32, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2))

        # Third layer
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # Fourth layer
        model.add(Dense(len(self.output_labels)))
        model.add(Activation('softmax'))

        # Model compiled, ready to train
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        # Assign model to object
        self.model = model

    def save_model(self):

        # Will overwrite previous model
        self.model.save(self.model_path)

if __name__ == '__main__':

    batch_size=1
    epochs=200
    validation_split=0.1


    label_dictionary = {'Matt':0, 'Ryan':1}
    labels = ['Matt', 'Ryan']

    nn = NeuralNetwork(input_shape=(129, 196), output_labels=labels)
    nn.create_model()
    nn.model.summary()

    audio_proc = processing.AudioProcessor()
    data_directory = 'labeled_wavs/'

    matt_dir = data_directory+labels[0]+'/'
    ryan_dir = data_directory+labels[1]+'/'

    matt_files = os.listdir(matt_dir)
    ryan_files = os.listdir(ryan_dir)

    print(matt_files)
    print(ryan_files)
    num_files = len(matt_files) + len(ryan_files)
    print(num_files)
    data_shape = (num_files, nn.input_shape[0], nn.input_shape[1])

    x_train = np.empty(shape=data_shape)
    y_train = np.empty(shape=num_files)
    i = 0
    for matt_wav in os.listdir(matt_dir):
        print(matt_wav)
        sample_freq, segment_times, spec = audio_proc.wav_to_spectrogram(wav_path=matt_dir+matt_wav)
        x_train[i] = spec
        y_train[i] = label_dictionary['Matt']
        i+=1
    for ryan_wav in os.listdir(ryan_dir):
        sample_freq, segment_times, spec = audio_proc.wav_to_spectrogram(wav_path=ryan_dir+ryan_wav)
        x_train[i] = spec
        y_train[i] = label_dictionary['Ryan']
        i+=1

    y_train = to_categorical(y_train)

    print(num_files*0.1)

    print(int((num_files*0.1)//batch_size))
    nn.model.fit(x_train, y_train, shuffle=True, epochs=epochs, validation_split=0.1, validation_steps=int((num_files*0.1)//batch_size), steps_per_epoch=int((num_files*0.9)//batch_size))

    nn.save_model()

    # Put spectrograms from labeled .wavs into
    # a training set, shuffle, and train
