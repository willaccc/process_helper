# data input format libraries
import pandas as pd
import numpy as np
# distribution check and stats test libraries
from scipy.stats import shapiro, anderson, normaltest, zscore
# visualization libraries
from matplotlib import pyplot as plt
import seaborn as sns


class OutlierDetect(object):

    def __init__(self,
                 data,
                 col_index=None):
        self.data = data
        self.col_index = col_index
        # overwrite data if column specified
        if col_index is not None:
            self.data = self.data[:, col_index]

    # data identification
    def _is_univariate(self,
                       data=None):
        """
        check if input is univariate
        if specified col_index list has only one element, return True
        :param data:
        :return: boolean
        """
        if data is None:
            data = self.data

        # get the shape of input data
        num_rows, num_cols = self.data.shape
        # check if it is univariate
        if num_cols == 1:
            return True
        # return True if specified column has only one element
        elif len(self.col_index) == 1:
            return True
        else:
            return False

    def _is_multivariate(self,
                         data=None):
        """
        check if input is multivariate
        if specified col_index list has more than one element, return True
        :param data:
        :return: boolean
        """
        if data is None:
            data = self.data

        # get the shape of input data
        num_rows, num_cols = data.shape
        # check if it is univariate
        if num_cols > 1:
            return True
        # return True if specified column has only one element
        elif len(self.col_index) > 1:
            return True
        else:
            return False

    # graphical check for distribution

    # normal distribution test

    # normal distribution
    def mean_z_score(self,
                     threshold=3,
                     data=None):
        """
        calculate z score using mean and std
        :param threshold: int, the boundary for outlier detection, default at 3
        :param data: numpy array, alternative data if data is not defined as self instance
        :return: list of index to indicate the location of outliers
        """
        # check if alternative data exists
        if data is None:
            z_score = list(zscore(self.data))
        else:
            z_score = list(zscore(data))

        # set up empty index list
        ind_list = []
        # iterate through list to identify z_score over threshold
        for index, element in enumerate(z_score):
            if element >= threshold:
                ind_list[index] = 1
            else:
                ind_list[index] = 0
        return ind_list

    # non parametric methods

    # IQR based method
    def iqr_base_range(self,
                       multiplier=1.5,
                       data=None):
        """
        identify inlier range using iqr based method
        :param multiplier:
        :param data: numpy array, alternative data if data is not defined as self instance
        :return: list of two elements, [lower limit, upper limit], obs outside of the limits will be outliers.
        """
        # check if alternative data exists
        if data is None:
            data = self.data
        else:
            pass

        qt_25 = np.percentile(data, 25)
        qt_75 = np.percentile(data, 75)
        # calculate iqr
        iqr = qt_75 - qt_25
        # calculate limit
        inlier_range = [qt_25 - multiplier * iqr, qt_75 + multiplier * iqr]
        return inlier_range

    # median-based z-score using MAD
    def median_z_score(self,
                       threshold,
                       quartile_value=0.6745,
                       data=None):
        """
        calculate z_score based on mad
        :param threshold:
        :param quartile_value: value for the xth quartile of the standard normal distribution, default at x=75,
        which is 0.6745
        :param data: numpy array, alternative data if data is not defined as self instance
        :return: list of index to indicate the location of outliers
        """
        # check if alternative data exists
        if data is None:
            data = self.data
        else:
            pass

        median = np.percentile(data, 50)
        diff = np.sum((data - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = list(quartile_value * diff / med_abs_deviation)

        ind_list = []
        # iterate through list to identify z_score over threshold
        for index, score in enumerate(modified_z_score):
            if score >= threshold:
                ind_list[index] = 1
            else:
                ind_list[index] = 0

        return ind_list

    # grubbs test
    def grubbs_test(self,
                    significance_level=0.05,
                    two_side=False,
                    data=None):
        """
        conduct grubbs test on current data set, can be called iteratively after removing the outliers
        :param significance_level:
        :param two_side:
        :return: boolean, if the max or the min is an outlier or not
        """
        # check for alternative data source
        if data is None:
            data = self.data
        else:
            pass
        # check if data is univariate
        assert self._is_univariate() is not True, "Input data is not univariate."

        # identify min and maximum
        min = np.min(data)
        max = np.max(data)
        # TODO: finish the grubb test
        # calculate grubb stats
        # grubb_stat =
        return 1

    # multivariate group

    # joint distribution check
    def joint_mean_z_score(self,
                           threshold=3,
                           data=None):
        """
        identify a joint distribution, identify outliers as outliers in both distributions.
        :param threshold:
        :param data:
        :return:
        """
        # check for alternative data source
        if data is None:
            data = self.data
        else:
            pass
        # check if data is multivariate
        assert self._is_multivariate() is not True, "Input data needs to be multivariate."

        # separate the input data
        data_one = data[:, 0]
        data_two = data[:, 1]

        # check mean_z_score for first distribution
        ind_list_one = self.mean_z_score(data=data_one, threshold=threshold)
        # check mean_z_score for second distribution
        ind_list_two = self.mean_z_score(data=data_two, threshold=threshold)
        # check overlapping index in both lists
        ind_joint = set(ind_list_one) & set(ind_list_two)

        return ind_joint


    # dbscan
    # local outlier
    # isolation forest

