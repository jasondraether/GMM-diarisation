from bayes_opt import BayesianOptimization
from model import GMMDiariser
from features import FeatureExtractor
import os
import numpy as np

# Define black box function to optimize
def run_instance(
    n_components,
    max_iter,
    emphasis_coefficient,
    energy_multiplier,
    energy_range,
    n_ccs,
    win_len,
    win_step,
    frame_length,
    frame_skip,
    top_db
):
    data_directory = 'profile_data/'
    X_train = []
    y_train = []

    # Instantiate model and feature extractor
    d_params = {'n_components':int(n_components), \
                'max_iter':int(max_iter)
    }
    diarizer = GMMDiariser(d_params)

    f_params = {'emphasis_coefficient':emphasis_coefficient, \
                'energy_multiplier':energy_multiplier, \
                'energy_range':int(energy_range), \
                'n_ccs':int(n_ccs), \
                'win_len':win_len, \
                'win_step':win_step, \
                'frame_length':int(frame_length), \
                'frame_skip':int(frame_skip), \
                'top_db':int(top_db)
    }
    extractor = FeatureExtractor(f_params)

    # Init diarizer with classes based on filesystem
    classes = os.listdir(data_directory)
    diarizer.init_profiles(labels=classes)
    D = len(classes)

    # Grab training and testing data
    # This only works when we concatenate data, if we don't we have to do a little extra
    for label in classes:
        class_dir = os.path.join(data_directory,label)

        X_class = extractor.extract_features_dir(dir=class_dir, concatenate=True)
        N_train = X_class.shape[0]
        y_class = diarizer.label_to_vector(label=label,N=N_train,D=D)

        X_train.append(X_class)
        y_train.append(y_class)

    # Flatten data
    X_train = np.concatenate(X_train,axis=0)
    y_train = np.concatenate(y_train,axis=0)

    # TODO: Remove this for bayesian tuning...
    X_train,y_train = diarizer.shuffle_data(X_train,y_train)

    # Do cross-validation
    accuracies = diarizer.cross_validate(X=X_train,y=y_train,n_folds=5,shuffle=False)
    return np.mean(accuracies)


# Note: the author probably doesn't support strict data types, so if
# this doesn't work, maybe add some rounding or data conversions...
pbounds = {
    'n_components':(1,32), \
    'max_iter':(100,5000), \
    'emphasis_coefficient':(0.9,0.99), \
    'energy_multiplier':(0.01,0.1), \
    'energy_range':(90,110), \
    'n_ccs':(2,20), \
    'win_len':(0.01,0.5), \
    'win_step':(0.01,0.25), \
    'frame_length':(256,1024), \
    'frame_skip':(128,512), \
    'top_db':(30,60)
}
optimizer = BayesianOptimization(
    f = run_instance,
    pbounds=pbounds,
    random_state=1
)

n_iter=10
init_points=10
optimizer.maximize(
    init_points=init_points,
    n_iter=n_iter
)

print(optimizer.max)
