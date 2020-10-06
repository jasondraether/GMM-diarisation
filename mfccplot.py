import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Input X is of shape (N, D) where N is the number of points and D is the dimension
# Input y is of shape (N, len(classes)), one-hot vector for classes
# Input labels is of shape (len(classes)), and contains the label indexed with argmax of y
# Output a DxD matrix of graphs
def plot_matrix(X, y, labels):
    N = X.shape[0]
    D = X.shape[1]
    n_classes = y.shape[1]
    columns = []
    for dim in range(D):
        columns.append('Feature {0}'.format(str(dim)))
    columns.append('Label')

    d_frames = []
    for class_id in range(n_classes):


        data_indices = np.argwhere(y[:,class_id] == 1).flatten()
        X_class = X[data_indices]

        n_class = X_class.shape[0]
        X_dataframe = np.zeros((n_class,D+1))
        X_dataframe[:,0:D] = X_class
        X_dataframe[:,D] = class_id

        d_frames.append(pd.DataFrame(X_dataframe,columns=columns))

    all_frames = pd.concat(d_frames)
    print('Plotting...')
    print(all_frames)
    sns.pairplot(all_frames.sample(50), hue='Label')
    plt.show()
