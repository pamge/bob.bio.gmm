#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>


import bob.core
import bob.io.base
import bob.learn.em

import numpy

from bob.bio.base.algorithm import Algorithm
from bob.bio.gmm.algorithm import GMM
import logging
logger = logging.getLogger("bob.bio.gmm")

class GMMSegment (GMM):
  """Algorithm for computing Universal Background Models and Gaussian Mixture Models of the features.
  Features must be normalized to zero mean and unit standard deviation."""

  def __init__(
      self,
      # parameters for the GMM
      number_of_gaussians,
      # parameters of UBM training
      kmeans_training_iterations = 25,   # Maximum number of iterations for K-Means
      gmm_training_iterations = 25,      # Maximum number of iterations for ML GMM Training
      training_threshold = 5e-4,         # Threshold to end the ML training
      variance_threshold = 5e-4,         # Minimum value that a variance can reach
      update_weights = True,
      update_means = True,
      update_variances = True,
      # parameters of the GMM enrollment
      relevance_factor = 4,         # Relevance factor as described in Reynolds paper
      gmm_enroll_iterations = 1,    # Number of iterations for the enrollment phase
      responsibility_threshold = 0, # If set, the weight of a particular Gaussian will at least be greater than this threshold. In the case the real weight is lower, the prior mean value will be used to estimate the current mean and variance.
      INIT_SEED = 5489,
      # scoring
      scoring_function = bob.learn.em.linear_scoring
  ):
    """Initializes the local UBM-GMM tool chain with the given file selector object"""

    # call base class constructor and register that this tool performs projection
    GMM.__init__(
        self,

        number_of_gaussians = number_of_gaussians,
        kmeans_training_iterations = kmeans_training_iterations,
        gmm_training_iterations = gmm_training_iterations,
        training_threshold = training_threshold,
        variance_threshold = variance_threshold,
        update_weights = update_weights,
        update_means = update_means,
        update_variances = update_variances,
        relevance_factor = relevance_factor,
        gmm_enroll_iterations = gmm_enroll_iterations,
        responsibility_threshold = responsibility_threshold,
        INIT_SEED = INIT_SEED,
        scoring_function = str(scoring_function),
    )

  def _check_feature(self, feature):
    """Checks that the features are appropriate"""
    if not isinstance(feature, numpy.ndarray) or feature.ndim != 2 or feature.dtype != numpy.float64:
      raise ValueError("The given feature is not appropriate")
    if self.ubm is not None and feature.shape[1] != self.ubm.shape[1]:
      raise ValueError("The given feature is expected to have %d elements, but it has %d" % (self.ubm.shape[1], feature.shape[1]))


  def project_ubm(self, array):
    features_per_seg = 250            # number of features per segment, the default value is 250, which means that if the window shift of mfcc is 10 ms, that means the duration of the segment is 2.5 s
    # array.shape[0] = number of MFCCs calculated on speech frame
    # array.shape[1] = dimension of MFCC features (ex. 60)
    # perform uniform linear segmentation on the feature sequence to obtain a segmented array
    segmented_array = [array[i:i+features_per_seg] for i in range(0, len(array), features_per_seg)]
    # loop on the segmented array to calculate the sufficient statistics on each segment
    gmm_stats_list = []
    for seg in segmented_array:
      logger.debug(" .... Projecting %d feature vectors" % seg.shape[0])                                                                          
      # Accumulates statistics
      gmm_stats = bob.learn.em.GMMStats(self.ubm.shape[0], self.ubm.shape[1])
      # self.ubm.shape[0]: number of gaussians
      # self.ubm.shape[1]: dimension of multivariate Gaussians, should be the same as the MFCC dimension (ex. if you're using MFCC-60, the dimension of the Gaussians should be 60)
      self.ubm.acc_statistics(seg, gmm_stats)
      gmm_stats_list.append(gmm_stats)
    # return the resulting statistics, gmm_stats_list contains a set of sufficient statistics belonging to each of the segments
    # Ex: let's suppose that the feature array is: array = [s_1, s_2, ..., s_T]
    #     then by performing a uniform segmentation we get the segmented array = [x_1, x_2, ..., x_M], with x_m = {s_k^m}, k=1,...,D. D being the number of features per segment (features_per_seg)
    #     in this case the returned gmm_stats_list is: [gmm_stats_1, gmm_stats_2, ..., gmm_stats_M]
    return gmm_stats_list


  def write_feature(self, gmm_stats_list, gmm_stats_file):
    """Saves GMM stats from returned list."""
    hdf5file = bob.io.base.HDF5File(gmm_stats_file, 'w')
    for i, g in enumerate(gmm_stats_list):
      groupname = '/segment_' + str(i)
      hdf5file.create_group(groupname)
      hdf5file.cd(groupname)
      g.save(hdf5file)
      hdf5file.cd('../')
    hdf5file.close()

  def project(self, feature):
    """Computes GMM statistics against a UBM, given an input 2D numpy.ndarray of feature vectors"""
    return self.project_ubm(feature)

  def read_gmm_stats(self, gmm_stats_file):
    """Reads GMM stats from file."""
    hdf5file = bob.io.base.HDF5File(gmm_stats_file)
    gmm_stats_list = []
    for grp in hdf5file.sub_groups():
      hdf5file.cd(grp)
      gmm_stats_list.append(bob.learn.em.GMMStats(hdf5file))
      hdf5file.cd('..')
    hdf5file.close()
    return gmm_stats_list