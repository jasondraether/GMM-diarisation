import numpy as np
import os

# Local modules
from model import SpeakerDiarizer
from features import FeatureExtractor

data_directory = 'profile_data/'
test_directory = 'test_data/'
X_train = []
y_train = []
X_test = []
y_test = []

# Instantiate model and feature extractor
d_params = {}
diarizer = SpeakerDiarizer(d_params)

f_params = {}
extractor = FeatureExtractor(f_params)

# Init diarizer with classes based on filesystem
classes = os.listdir(data_directory)
diarizer.init_profiles(labels=classes)
D = len(classes)

print('Parsing training and testing data.')
# Grab training and testing data
# This only works when we concatenate data, if we don't we have to do a little extra
for label in classes:
    class_dir = os.path.join(data_directory,label)
    class_test_dir = os.path.join(test_directory,label)
    X_class = extractor.extract_features_dir(dir=class_dir, concatenate=True)
    X_class_test = extractor.extract_features_dir(dir=class_test_dir,concatenate=True)
    N_train = X_class.shape[0]
    N_test = X_class_test.shape[0]
    y_class = diarizer.label_to_vector(label=label,N=N_train,D=D)
    y_class_test = diarizer.label_to_vector(label=label,N=N_test,D=D)
    X_train.append(X_class)
    y_train.append(y_class)
    X_test.append(X_class_test)
    y_test.append(y_class_test)

# Flatten data
X_train = np.concatenate(X_train,axis=0)
y_train = np.concatenate(y_train,axis=0)
X_test = np.concatenate(X_test,axis=0)
y_test = np.concatenate(y_test,axis=0)
n_train = y_train.shape[0]
n_test = y_test.shape[0]

print('Training with {0} data points, testing with {1} data points'.format(n_train,n_test))

print('Tuning parameters with cross-validation')
X_train,y_train = diarizer.shuffle_data(X_train,y_train)
# Do cross-validation, then test on hidden set
# for n_components in range(1,32):
#     d_params['n_components'] = n_components
#     diarizer.set_params(d_params)
#     accuracies = diarizer.cross_validate(X=X_train,y=y_train,n_folds=5,shuffle=False)
#     print(n_components,accuracies,np.mean(accuracies))

print('Testing')
diarizer.train(X=X_train,y=y_train)
acc = diarizer.test(X=X_test,y=y_test)
print(acc)
