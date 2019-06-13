from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras import regularizers, optimizers
import AudioProcessor

'''
Neural network code for
audio-learning
'''
class NeuralNetwork:

    def __init__(self, kernel_size, activation, dropout, pool_size, input_shape, output_labels, learning_rate, decay):
        self.kernel_size = kernel_size
        self.activation = activation
        self.dropout = dropout
        self.pool_size = pool_size
        self.input_shape = input_shape # Shape of spectrogram is (129, 196)
        self.output_labels = output_labels
        self.learning_rate = learning_rate
        self.decay = decay

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=self.kernel_size, activation=self.activation,
            input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=self.pool_size))
        model.add(Dropout(self.dropout))
        model.add(Conv2D(64, kernel_size=self.kernel_size, activation=self.activation,
            input_shape=self.input_shape))
        model.add(Dropout(self.dropout))
        model.add(Flatten())
        model.add(Dense(32))
        model.add(Dense(1, activation='softmax'))
        model.compile(optimizers.rmsprop(lr=self.learning_rate, decay=self.decay), loss="categorical_crossentropy",metrics=["accuracy"])

        self.model = model

if __name__ == '__main__':

    nn = NeuralNetwork((3,3), 'relu', 0.5, (2,2), (129, 196, 1), ['Matt', 'Ryan'], 0.0005, 1e-6)
    nn.create_model()
    nn.model.summary()

    audio_proc = AudioProcessor()

    # Put spectrograms from labeled .wavs into
    # a training set, shuffle, and train
