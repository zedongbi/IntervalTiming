import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import os
import pickle
from scipy.stats import ortho_group
from mpl_toolkits.mplot3d import Axes3D

import sys

sys.path.append("..")

_color_list = ['blue', 'red', 'black', 'yellow']
fs = 12 # fontsize


class PCASpace(object):
    """
    data,  list of numpy array of shape (xtime, rnn_size),
    Note the xtime of elements in the list may not be the same.
    The number of elements in the list is the number of parameter conditions
    pca_n, int, the number of PCA dimensions need to be studied.
    """
    def __init__(self, data, pca_n, model_dir):
        self.data = data
        self.pca_n = pca_n

        # perform PCA
        X = np.concatenate(tuple(data), axis=0)
        self.pca = PCA(n_components=pca_n)
        self.pca.fit(X)
        X_pca = self.pca.transform(X)
        data_pca = list()
        pos = 0
        for y in data:
            data_pca.append(X_pca[pos:pos+y.shape[0]])
            pos = pos + y.shape[0]
        self.data_pca = data_pca

        self.X_pca = X_pca

        # (rnn_n, pca_n)
        self.pca_transform_matrix = self.pca.components_.T

        readme = 'data_original: original data. List of numpy array of shape (xtime, rnn_size). Length of list is the' \
                 'number of parameter conditions. \n' + \
                 'data_pca: data in pca space. List of numpy array of shape (xtime, pca_n). Length of list is the number' \
                 'of parameter conditions. \n' + \
                 'pca_transform_matrix: transform matrix to pca space. Numpy array of shape (rnn_n, pca_n). \n'

        #print(self.data[0].shape)
        #print(self.data_pca[0].shape)

        self.result = {
            'data_original': self.data,
            'data_pca': self.data_pca,
            'pca_transform_matrix': self.pca_transform_matrix,
            'readme': readme
                  }
        save_dir_name = os.path.join(model_dir, 'pca_space' + '.pkl')
        #print(save_dir_name)
        '''
        save_dir_name = os.path.join(model_dir, 'pca_space' + '.pkl')
        print('PCA_space_transform saved at {:s}'.format(save_dir_name))
        with open(save_dir_name, 'wb') as f:
            pickle.dump(result, f)
        '''