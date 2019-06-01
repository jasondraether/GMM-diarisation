from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

'''
Neural network code for
audio-learning
'''
class NeuralNetwork:

    def __init__(self, kernel_size, activation, dropout, pool_size, input_shape, output_labels):
        self.kernel_size = kernel_size
        self.activation = activation
        self.dropout = dropout
        self.pool_size = pool_size
        self.input_shape = input_shape # Shape of spectrogram is (129, 196)
        self.output_labels = output_labels

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
        model.add(Dense(32, self.activation))
        model.add(Dense(1, activation='softmax'))

        self.model = model
