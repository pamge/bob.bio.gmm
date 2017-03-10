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

class IVectorSegment (IVector, GMMSegment):
  """Tool for extracting I-Vectors"""

  def __init__(
      self,
      # IVector training
      subspace_dimension_of_t,       # T subspace dimension
      tv_training_iterations = 25,   # Number of EM iterations for the JFA training
      update_sigma = True,
      use_whitening = True,
      use_lda = False,
      use_wccn = False,
      use_plda = False,
      lda_dim = 50,
      plda_dim_F  = 50,
      plda_dim_G = 50,
      plda_training_iterations = 50,
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

  def train_ivector(self, training_stats):
    logger.info("  -> Training IVector enroller")
    self.tv = bob.learn.em.IVectorMachine(self.ubm, self.subspace_dimension_of_t, self.variance_threshold)

    # train IVector model
    bob.learn.em.train(self.ivector_trainer, self.tv, training_stats, self.tv_training_iterations, rng=self.rng)

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


  def save_projector(self, projector_file):
    # Save the IVector base AND the UBM AND the whitening into the same file
    hdf5file = bob.io.base.HDF5File(projector_file, "w")
    hdf5file.create_group('Projector')
    hdf5file.cd('Projector')
    self.save_ubm(hdf5file)

    hdf5file.cd('/')
    hdf5file.create_group('Enroller')
    hdf5file.cd('Enroller')
    self.tv.save(hdf5file)

    if self.use_whitening:
      hdf5file.cd('/')
      hdf5file.create_group('Whitener')
      hdf5file.cd('Whitener')
      self.whitener.save(hdf5file)
    
    if self.use_lda:
      hdf5file.cd('/')
      hdf5file.create_group('LDA')
      hdf5file.cd('LDA')
      self.lda.save(hdf5file)

    if self.use_wccn:
      hdf5file.cd('/')
      hdf5file.create_group('WCCN')
      hdf5file.cd('WCCN')
      self.wccn.save(hdf5file)
            
    if self.use_plda:
      hdf5file.cd('/')
      hdf5file.create_group('PLDA')
      hdf5file.cd('PLDA')
      self.plda_base.save(hdf5file)
      

  def load_tv(self, tv_file):
    hdf5file = bob.io.base.HDF5File(tv_file)
    self.tv = bob.learn.em.IVectorMachine(hdf5file)
    # add UBM model from base class
    self.tv.ubm = self.ubm
    
  def load_projector(self, projector_file):
    """Load the GMM and the ISV model from the same HDF5 file"""
    hdf5file = bob.io.base.HDF5File(projector_file)

    # Load Projector
    hdf5file.cd('/Projector')
    self.load_ubm(hdf5file)

    # Load Enroller
    hdf5file.cd('/Enroller')
    self.load_tv(hdf5file)

    if self.use_whitening:
      # Load Whitening
      hdf5file.cd('/Whitener')
      self.load_whitener(hdf5file)
    
    if self.use_lda:
      # Load LDA
      hdf5file.cd('/LDA')
      self.load_lda(hdf5file)
    
    if self.use_wccn:    
      # Load WCCN
      hdf5file.cd('/WCCN')
      self.load_wccn(hdf5file)

    if self.use_plda:   
     # Load PLDA
      hdf5file.cd('/PLDA')
      self.load_plda(hdf5file)


  def project_ivector(self, gmm_stats_list):
    tv_project = []
    #import ipdb
    #ipdb.set_trace()
    for gmm_stats in gmm_stats_list:
      tv_project.append(self.tv.project(gmm_stats))
    return tv_project

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
  def write_feature(self, data, feature_file):
    """Saves the feature, which is the (whitened) I-Vector."""
    bob.bio.base.save(data, feature_file)

  def read_feature(self, feature_file):
    """Read the type of features that we require, namely i-vectors (stored as simple numpy arrays)"""
    return bob.bio.base.load(feature_file)

  def score(self, model, probe):
    print 'no scoring'
