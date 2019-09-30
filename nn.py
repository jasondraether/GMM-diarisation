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

    def __init__(self, input_shape=(0, 0), output_labels=None, learning_rate=0.001, decay=0.001, model_path='models/nn.model', weight_path='models/nn-weights.h5', model=None):
        self.input_shape = input_shape # Shape of spectrogram is (129, 71) NOTE: This may change based on sampling rates and such. Make sure to have static numbers!
        self.output_labels = output_labels # Classes
        self.learning_rate = learning_rate # Unused
        self.decay = decay # Unused
        self.model = model # Compiled model
        self.model_path = model_path # Path to save model
        self.weight_path = weight_path # I think this is redundant, but it doesn't hurt to have it implemented
        self.label_dictionary = {label:index for index,label in enumerate(self.output_labels)}

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
        model.add(Dense(len(self.output_labels))) # This should be two always
        model.add(Activation('softmax'))

        # model compiled, ready to train
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        # Assign model to object
        self.model = model

        # TODO: Add cascaded models, use catagorical_crossentropy for multi-class

    def save_model(self):
        # Will overwrite previous model
        self.model.save(self.model_path)
        self.model.save_weights(self.weight_path)

    def load_model(self):
        self.model = load_model(self.model_path)
        self.model.load_weights(self.weight_path)

    def train_model(self, data_directory, batch_size, epochs, validation_split):

        audio_proc = processing.AudioProcessor()
        training_directories = [data_directory+label+'/' for label in labels]
        training_files = [os.listdir(dir) for dir in training_directories] # Should be 2D
        num_files = sum([len(file) for file in training_files])

        x_train = []
        y_train = []

        for label in labels:
            wav_paths = os.listdir(data_directory+label)
            x_train += [audio_proc.wav_to_spectrogram(wav_path=data_directory+label+'/'+class_wav) for class_wav in wav_paths]
            y_train += [self.label_dictionary[label] for x in range(0, len(wav_paths))]

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        y_train = to_categorical(y_train)

        nn.model.fit(x_train, y_train, shuffle=True, epochs=epochs, validation_split=validation_split, validation_steps=int((num_files*validation_split)//batch_size), steps_per_epoch=int((num_files*(1-validation_split))//batch_size))
        self.save_model()

    def test_model(self, test_directory):
        self.load_model()
        audio_proc = processing.AudioProcessor()
        filenames = [test_directory+test_wav for test_wav in os.listdir(test_directory)]
        x_test = [audio_proc.wav_to_spectrogram(wav_path=test_directory+test_wav) for test_wav in filenames]
        predictions = self.model.predict(x_test)
        prediction_data = list(zip(predictions,filenames))
        with open('predictions.txt', 'w') as f:
            f.write("=== List of predictions with prediction and filename ===\n")
            f.write("Using label dictionary:\n")
            for key, value in self.label_dictionary.items():
                f.write("%s\n" % (key,value)) # Write integer labels with text labels
            f.write("\nPredictions:\n")
            for pred in prediction_data:
                f.write("%s\n" % pred) # Write text file of prediction with filename

if __name__ == '__main__':

    test = 0 # 0 if training, 1 if testing

    batch_size=5
    epochs=300
    validation_split=0.1
    data_directory='labeled_wavs/'
    input_shape=(129, 71) # 16 kHz, 16-bit PCM encoding, 1 second long
    labels = ['Matt', 'Ryan']

    if test:
        nn = NeuralNetwork(output_labels=labels)
        nn.load_model()
        nn.model.summary()
        nn.test_model('test_data/')
    else:
        nn = NeuralNetwork(input_shape=input_shape, output_labels=labels)
        nn.create_model()
        nn.model.summary()
        nn.train_model(data_directory=data_directory, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
