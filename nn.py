from keras.models import Sequential, load_model
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

    def load_model(self):
        self.model = load_model(self.model_path)

    def train_model(self, data_directory, batch_size, epochs, validation_split):
        self.label_dictionary = {label:index for index,label in enumerate(self.output_labels)}
        audio_proc = processing.AudioProcessor()
        training_directories = [data_directory+label+'/' for label in labels]
        training_files = [os.listdir(dir) for dir in training_directories] # Should be 2D
        num_files = sum(len(file) for file in training_files)

        x_train = []
        y_train = []

        x_train += [audio_proc.wav_to_spectrogram(wav_path=matt_dir+matt_wav) for matt_wav in os.listdir(matt_dir)]
        y_train += [label_dictionary['Matt'] for x in range(0, len(x_train))

        x_train += [audio_proc.wav_to_spectrogram(wav_path=ryan_dir+ryan_wav) for ryan_wav in os.listdir(ryan_dir)]
        y_train += [label_dictionary['Ryan'] for x in range(0, len(x_train))

        y_train = to_categorical(y_train)

        nn.model.fit(x_train, y_train, shuffle=True, epochs=epochs, validation_split=validation_split, validation_steps=int((num_files*validation_split)//batch_size), steps_per_epoch=int((num_files*(1-validation_split))//batch_size))
        self.save_model()

    def test_model(self, test_directory):
        self.load_model()
        audio_proc = processing.AudioProcessor()
        x_test = [audio_proc.wav_to_spectrogram(wav_path=test_directory) for test_wav in os.listdir(test_directory)]
        predictions = self.model.predict(test_x)
        with open('predictions.txt', 'w') as f:
            for pred in predictions:
                f.write("%s\n" % pred)


if __name__ == '__main__':

    batch_size=50
    epochs=200
    validation_split=0.1
    data_directory='labeled_wavs/'
    input_shape=(129, 196) # Find a way to implement this better?
    labels = ['Matt', 'Ryan']

    nn = NeuralNetwork(input_shape=input_shape, output_labels=labels)
    nn.create_model()
    nn.model.summary()
    nn.train_model(data_directory=data_directory, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
