import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import os
import pickle
from scipy.stats import ortho_group

from scipy import optimize


_color_list = ['blue', 'red', 'black', 'yellow']
fs = 12 # fontsize

'''
    total_mean = np.mean(projection)
    total_var = np.sum(np.square(projection - total_mean))

    explained_mean = np.mean(projection, axis=0)
    explained_var = np.sum(np.square(projection - np.reshape(explained_mean, (1, projection.shape[1]))))

    return 1 - explained_var/total_var
'''


def scaling_component(data_rescale):
    '''
    :param data_rescale: dimension: (parameter_n, ref_time, pca_n)
    :return:
    '''
    parameter_n, ref_time, pca_n = data_rescale.shape

    def _cost_func(kernel):
        kernel = kernel/np.linalg.norm(kernel)  # method to initialize kernel

        projection = np.matmul(data_rescale, kernel)
        # dimension: (parameter_n, ref_time)
        explained_mean = np.mean(projection, axis=0)
        explained_var = np.sum(np.square(projection - np.reshape(explained_mean, (1, projection.shape[1]))))

        total_mean = np.mean(projection)
        total_var = np.sum(np.square(projection - total_mean))

        return explained_var/total_var

    while True:
        try:
            # Generate initialization matrix
            kernel_initial = np.random.uniform(-1, 1, pca_n)

            res = optimize.minimize(_cost_func, x0=kernel_initial, method='BFGS')
            break
        except RuntimeWarning:
            print('catch')
            continue
    #print(_cost_func(res.x))

    return res.x/np.linalg.norm(res.x)


def scaling_component_2(data_rescale):
    '''
    :param data_rescale: dimension: (parameter_n, ref_time, pca_n)
    :return:
    '''
    parameter_n, ref_time, pca_n = data_rescale.shape

    def _cost_func(kernel):
        kernel = kernel/np.linalg.norm(kernel)  # method to initialize kernel

        projection = np.matmul(data_rescale, kernel)
        # dimension: (parameter_n, ref_time)
        var = np.var(projection, axis=0)
        cost = np.mean(var)
        return cost

    # Generate initialization matrix
    kernel_initial = np.random.uniform(-1, 1, pca_n)

    res = optimize.minimize(_cost_func, x0=kernel_initial, method='BFGS')

    #print(_cost_func(res.x))

    return res.x/np.linalg.norm(res.x)


class ScalingSpace(object):
    """
    data_pca,  list of numpy array of shape (xtime, pca_n),
    Note the xtime of elements in the list may not be the same.
    The number of elements in the list is the number of parameter conditions
    save_dir_name, string, the name of file to which the results to be saved
    """
    def __init__(self, data_pca, save_dir_name):
        self.data_pca = data_pca

        # rescale time of data_pca
        maxT = max([x.shape[0] for x in data_pca])
        ref_time = np.array(range(maxT))/(maxT-1)

        # dimension: (parameter_n, ref_time, pca_n)
        data_rescale = np.zeros((len(data_pca), ref_time.shape[0], data_pca[0].shape[1]))

        for para_idx in range(len(data_pca)):
            x = data_pca[para_idx]
            curr_time = np.array(range(x.shape[0]))/(x.shape[0]-1)
            for pca_idx in range(x.shape[1]):
                data_rescale[para_idx, :, pca_idx] = np.interp(ref_time, curr_time, x[:, pca_idx])
        self.data_rescale = data_rescale

        (self.parameter_n, self.ref_time_n, self.pca_n) = self.data_rescale.shape

        # (rnn_n, pca_n)
        self.scaling_transform_matrix = self.transform()

        readme = 'data_pca: data in pca space. List of numpy array of shape (xtime, pca_n). Length of list is the number' \
                 'of parameter conditions. \n' + \
                 'data_rescale: data in pca space of rescaled time. Numpy array of shape (parameter_n, ref_time_n, pca_n),' \
                 'with ref_time_n being the number of time points in reference trajectory. \n' + \
                 'scaling_transform_matrix: transform matrix from pca space to scaling space. Numpy array of shape (pca_n, pca_n).'

        self.result = {
            'data_pca': self.data_pca,
            'data_rescale': self.data_rescale,
            'scaling_transform_matrix': self.scaling_transform_matrix,
            'readme': readme
                  }
        '''
        print('scaling_space_transform saved at {:s}'.format(save_dir_name))
        with open(save_dir_name, 'wb') as f:
            pickle.dump(result, f)
        '''

    def transform(self):
        """
        transform self.data_rescale to scaling space

        save_dir: directory to save the result
        :return:
        """

        parameter_n, ref_time_n, pca_n = (self.parameter_n, self.ref_time_n, self.pca_n)

        '''
        (parameter_n, ref_time, pca_n) = self.data_rescale.shape
        plt.figure(1)
        for i in range(parameter_n):
            plt.plot(np.array(range(len(input_data[0, :, i]))), input_data[0, :, i])
        plt.show()
        '''
        transform_matrices = list()

        input_data = self.data_rescale

        for axis_idx in range(0, self.pca_n):

            kernel = scaling_component(input_data[:, :, axis_idx:])

            if self.pca_n - axis_idx > 1:

                temp_transform = ortho_group.rvs(self.pca_n - axis_idx)
                temp_transform[:, 0] = kernel
                q, _ = np.linalg.qr(temp_transform)

                s = np.identity(self.pca_n)
                s[axis_idx:, axis_idx:] = q
                transform_matrices.append(s)
            else:
                s = np.identity(self.pca_n)
                transform_matrices.append(s)

            input_data_reshape = np.reshape(input_data, (-1, self.pca_n))
            input_data = np.matmul(input_data_reshape, s).reshape(parameter_n, ref_time_n, pca_n)

            #print(input_data.shape)

        transform_matrix = np.identity(pca_n)
        for x in transform_matrices:
            transform_matrix = np.matmul(transform_matrix, x)

        return transform_matrix


def variance_explained(data, transform_matrix):
    """
    calculate the ratio of variance explained by different directions
    :param data: numpy array of shape (n_point, pca_n)
    :param transform_matrix: numpy of shape (pca_n, pca_n)
    :return: numpy of shape(1, pca_n)
    """
    transformed_data = np.matmul(data, transform_matrix)
    total_var = np.sum(np.var(transformed_data, axis=0))
    var_seq_axes = np.var(transformed_data, axis=0)

    return var_seq_axes/total_var


def scaling_index(data_rescale, kernel):
    """
    Quantifies the scaling activity of the projection of data along the vector defined by kernel.
    sequential index is defined as:
    (variance around the mean activity over different parameter conditions, average over time)/(variance over all data)
    :param data_rescale: time-rescaled data, numpy array of shape (parameter_n, ref_time_n, pca_n)
    :param kernel: the projection of data along the vector, numpy array of vector of length pca_n
    :return: float
    """
    parameter_n, ref_time_n, pca_n = data_rescale.shape
    input_data_reshape = np.reshape(data_rescale, (-1, pca_n))
    kernel_reshape = np.reshape(kernel, (pca_n, 1))

    # (parameter_n, time_n)
    projection = np.matmul(input_data_reshape, kernel_reshape).reshape(parameter_n, ref_time_n)

    total_mean = np.mean(projection)
    #print(total_mean)

    total_var = np.sum(np.square(projection - total_mean))
    #print('total_var', total_var)

    explained_mean = np.mean(projection, axis=0)
    explained_var = np.sum(np.square(projection - np.reshape(explained_mean, (1, projection.shape[1]))))

    return 1 - explained_var/total_var


def scaling_index_multi_dim(data_rescale, kernel):
    """
    Quantifies the scaling activity of the projection of data along the vector defined by kernel.
    sequential index is defined as:
    (variance around the mean activity over different parameter conditions, average over time)/(variance over all data)
    :param data_rescale: time-rescaled data, numpy array of shape (parameter_n, ref_time_n, pca_n)
    :param kernel: the projection of data along the vector, numpy array of vector of length pca_n
    :return: float
    """
    #parameter_n, ref_time_n, pca_n = data_rescale.shape
    #pca_n, sca_n = kernel.shape
    #input_data_reshape = np.reshape(data_rescale, (-1, pca_n))
    #kernel_reshape = np.reshape(kernel, (pca_n, 1))

    # (parameter_n, time_n, sca_n)
    projection = np.matmul(data_rescale, kernel)
    total_mean = np.mean(projection, axis=(0, 1))[np.newaxis, np.newaxis, :]

    #print('total_mean', total_mean)
    total_var = np.sum(np.square(projection - total_mean))
    #print('total_var', total_var)

    explained_mean = np.mean(projection, axis=0)[np.newaxis, :, :]
    explained_var = np.sum(np.square(projection - explained_mean))

    return 1 - explained_var/total_var
