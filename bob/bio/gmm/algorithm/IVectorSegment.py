#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

import bob.core
import bob.io.base
import bob.learn.linear
import bob.learn.em

import numpy
from .IVector import IVector
from .GMMSegment import GMMSegment
from bob.bio.base.algorithm import Algorithm

import logging
logger = logging.getLogger("bob.bio.gmm")

class IVectorSegment (GMMSegment, IVector):
  """Tool for extracting I-Vectors"""

  def __init__(
      self,
      # IVector training
      subspace_dimension_of_t,       # T subspace dimension
      tv_training_iterations = 3,   # Number of EM iterations for the TV training
      update_sigma = True,
      use_whitening = False,
      use_lda = False,
      use_wccn = False,
      use_plda = False,
      lda_dim = 2,
      plda_dim_F  = 2,
      plda_dim_G = 2,
      plda_training_iterations = 2,
      # parameters of the GMM
      **kwargs
  ):
    """Initializes the local GMM tool with the given file selector object"""
    # call base class constructor with its set of parameters
    GMMSegment.__init__(self, **kwargs)

    # call tool constructor to overwrite what was set before
    Algorithm.__init__(
        self,
        performs_projection = True,
        use_projected_features_for_enrollment = True,
        requires_enroller_training = False, # not needed anymore because it's done while training the projector
        split_training_features_by_client = True,

        subspace_dimension_of_t = subspace_dimension_of_t,
        tv_training_iterations = tv_training_iterations,
        update_sigma = update_sigma,
        use_whitening = use_whitening,
        use_lda = use_lda,
        use_wccn = use_wccn,
        use_plda = use_plda,
        lda_dim = lda_dim,
        plda_dim_F  = plda_dim_F,
        plda_dim_G = plda_dim_G,
        plda_training_iterations = plda_training_iterations,

        multiple_model_scoring = None,
        multiple_probe_scoring = None,
        **kwargs
    )

    self.update_sigma = update_sigma
    self.use_whitening = use_whitening
    self.use_lda = use_lda
    self.use_wccn = use_wccn
    self.use_plda = use_plda
    self.subspace_dimension_of_t = subspace_dimension_of_t
    self.tv_training_iterations = tv_training_iterations
    
    self.ivector_trainer = bob.learn.em.IVectorTrainer(update_sigma=update_sigma)
    self.whitening_trainer = bob.learn.linear.WhiteningTrainer()
    
    self.lda_dim = lda_dim
    self.lda_trainer = bob.learn.linear.FisherLDATrainer(strip_to_rank=False)
    self.wccn_trainer = bob.learn.linear.WCCNTrainer()
    self.plda_trainer = bob.learn.em.PLDATrainer()
    self.plda_dim_F  = plda_dim_F
    self.plda_dim_G = plda_dim_G
    self.plda_training_iterations = plda_training_iterations


  def _check_ivector(self, feature):
    """Checks that the features are appropriate"""
    if not isinstance(feature, numpy.ndarray) or feature.dtype != numpy.float64:
      raise ValueError("The given feature is not appropriate")

  def train_projector(self, train_features, projector_file):
    """Train Projector and Enroller at the same time"""
    
    [self._check_feature(feature) for client in train_features for feature in client]
    train_features_flatten = [feature for client in train_features for feature in client]

    # train UBM
    data = numpy.vstack(train_features_flatten)
    self.train_ubm(data)
    del data

    # project training data
    logger.info("  -> Projecting training data")
    train_gmm_stats = [[GMMSegment.project_ubm(self, feature) for feature in client] for client in train_features]
    train_gmm_stats_flatten = [stats for client in train_gmm_stats for segments in client for stats in segments]
    

    # train IVector
    logger.info("  -> Projecting training data")
    self.train_ivector(train_gmm_stats_flatten)

    # project training i-vectors
    train_ivectors = [[self.project_ivector(stats) for stats in client] for client in train_gmm_stats]
    train_ivectors_flatten = [stats for client in train_ivectors for segments in client for stats in segments]

    if self.use_whitening:
      # Train Whitening
      self.train_whitener(train_ivectors_flatten)
      # whitening and length-normalizing i-vectors
      train_ivectors = [[self.project_whitening(ivec) for ivec in client] for client in train_ivectors]
    
    if self.use_lda:
      self.train_lda(train_ivectors)
      train_ivectors = [[self.project_lda(ivec) for ivec in client] for client in train_ivectors]
      
    if self.use_wccn:
      self.train_wccn(train_ivectors)
      train_ivectors = [[self.project_wccn(ivec) for ivec in client] for client in train_ivectors]
      
    if self.use_plda:
      self.train_plda(train_ivectors)

    # save
    self.save_projector(projector_file)


  def project_ivector(self, gmm_stats_list):
    tv_project = []
    for gmm_stats in gmm_stats_list:
      tv_project.append(self.tv.project(gmm_stats))
    return tv_project

  def project_whitening(self, ivector):
    whitened_list = []
    for ivec in ivector:
      whitened = self.whitener.forward(ivec)
      whitened_list.append(whitened / numpy.linalg.norm(whitened))
    return whitened_list
  
  def project_lda(self, ivector):
    out_ivector_list = []
    for ivec in ivector:
      out_ivector = numpy.ndarray(self.lda.shape[1], numpy.float64)
      self.lda(ivec, out_ivector)
      out_ivector_list.append(out_ivector)
    return out_ivector_list

  def project_wccn(self, ivector):
    out_ivector_list = []
    for ivec in ivector:
      out_ivector = numpy.ndarray(self.wccn.shape[1], numpy.float64)
      self.wccn(ivec, out_ivector)
      out_ivector_list.append(out_ivector)
    return out_ivector_list

  #######################################################
  ############## IVector projection #####################
  def project(self, feature_array):
    """Computes GMM statistics against a UBM, then corresponding Ux vector"""
    self._check_feature(feature_array)
    # project UBM
    projected_ubm = GMMSegment.project_ubm(self, feature_array)
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
    return ivector

  #######################################################
  ################## Read / Write I-Vectors ####################

  def score(self, model, probe):
    print('no scoring')
