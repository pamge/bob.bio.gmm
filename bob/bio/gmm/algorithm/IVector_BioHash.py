#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Vedrana Krivokuca

from bob.bio.gmm.algorithm.IVector import IVector
from bob.btp import biohash
import numpy
import bob.bio.base
import scipy

def load_algorithm(algorithm):
    if algorithm is None:
        raise ValueError("Please provide the name of a valid algorithm.")
    elif isinstance(algorithm, str):
        algorithm = bob.bio.base.load_resource(algorithm, 'algorithm')
    elif isinstance(algorithm, Algorithm):
        algorithm = algorithm
    else:
        raise ValueError("The provided algorithm type is not understood.")

    return algorithm


class IVector_BioHash (IVector):
  """Protect IVectors using BioHashing"""

  def __init__(self, subspace_dimension_of_t = None, use_lda = None, use_wccn = None, use_plda = None, length = None, user_seed = None, **kwargs):
    # Initializing attributes of parent class, IVector
    IVector.__init__(
      self, 
      requires_seed = True, 
      subspace_dimension_of_t = subspace_dimension_of_t, # dim of ivector - try setting to 400
      update_sigma = True,
      tv_training_iterations = 25,  # Number of EM iterations for the TV training
      number_of_gaussians = 256, 
      training_threshold = 0.0, 
      use_lda = use_lda, # F
      use_wccn = use_wccn, # F
      use_plda = use_plda, # F
      lda_dim = 50,
      plda_dim_F = 10, # remove
      plda_dim_G = 50,
      plda_training_iterations = 200, # remove
      **kwargs)
    # Initializing attributes of child class, IVector_BioHash
    self.length = length
    self.user_seed = user_seed


  def protect(self, ivector, user_seed):
    # THIS METHOD IS DEFINED IN THE CHILD CLASS, IVector_BioHash, AND NOT IN THE PARENT CLASS, IVector
    """ Creates BioHash by projecting the ivector onto a set of randomly generated basis vectors and then binarising the resulting vector. """
    if self.user_seed == None: # normal scenario, so user_seed = client_id
        print "NORMAL scenario user seed: %s\n" % (user_seed)
        return biohash.create_biohash(ivector, self.length, user_seed)
    else: # stolen token scenario, so user_seed will be some randomly generated number (same for every person in the database), specified in config file
        print "STOLEN TOKEN scenario user seed: %s\n" % (self.user_seed)
        return biohash.create_biohash(self.ivector, self.length, self.user_seed)


  def project(self, feature_array, user_seed):
    # OVERRIDING "project" method from parent class, IVector
    """Computes GMM statistics against a UBM, then corresponding Ux vector, then secures ivector using BioHash"""
    self._check_feature(feature_array)
    # project UBM
    projected_ubm = self.project_ubm(feature_array)
    # project I-Vector
    ivector = self.project_ivector(projected_ubm)
    # whiten I-Vector
    if self.use_whitening:
      ivector = self.project_whitening(ivector)
    # LDA projection
    if self.use_lda:
      ivector = self.project_lda(ivector)
    # WCCN projection
    if self.use_wccn:
      ivector = self.project_wccn(ivector)
    # Protection via BioHashing
    return self.protect(ivector, user_seed)


  def enroll(self, enroll_features):
    # OVERRIDING "enroll" method from parent class, IVector
    """Performs BioHash IVector enrollment"""
    average_biohash = scipy.stats.mode(numpy.vstack(enroll_features), axis=0)[0][0] # finds most common bit in each column of stacked BioHashes
    return average_biohash


  def read_model(self, model_file):
    # OVERRIDING "read_model" method from parent class, IVector
    """Reads the BioHashed i-vector that holds the model"""
    return bob.bio.base.load(model_file)


  def score(self, ref_biohash, probe_biohash):
    # OVERRIDING "score" method from parent class, IVector
    """Computes the Hamming distance between the given BioHash model and the given BioHash probe."""
    print "r: %s\np: %s\nHamming distance = %s" % (ref_biohash, probe_biohash, biohash.calc_hamm_dist(ref_biohash, probe_biohash))
    return -1 * biohash.calc_hamm_dist(ref_biohash, probe_biohash) # this is a distance measure (as opposed to a similarity measure), so multiply result by -1


  def score_for_multiple_probes(self, ref_biohash, probe_biohashes):
    # OVERRIDING "score_for_multiple_probes" method from parent class, IVector
    """This function computes the score between the given BioHash model and several given BioHash probe files."""
    probe = scipy.stats.mode(numpy.vstack(enroll_features), axis=0)[0][0] # finds most common bit in each column of stacked BioHash probes
    return self.score(ref_biohash, probe)