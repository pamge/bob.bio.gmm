import logging
logger = logging.getLogger("bob.bio.gmm")

import bob.io.base
import os
import shutil

from bob.bio.base.tools.FileSelector import FileSelector
from bob.bio.base import utils, tools

def train_isv(algorithm, force=False):
  """Finally, the UBM is used to train the ISV projector/enroller."""
  fs = FileSelector.instance()

  if utils.check_file(fs.projector_file, force, 800):
    logger.info("ISV training: Skipping ISV training since '%s' already exists", fs.projector_file)
  else:
    # read UBM into the ISV class
    algorithm.load_ubm(fs.ubm_file)

    # read training data
    training_list = fs.training_list('projected_gmm', 'train_projector', arrange_by_client = True)
    train_gmm_stats = [[algorithm.read_gmm_stats(filename) for filename in client_files] for client_files in training_list]

    # perform ISV training
    logger.info("ISV training: training ISV with %d clients", len(train_gmm_stats))
    algorithm.train_isv(train_gmm_stats)
    # save result
    bob.io.base.create_directories_safe(os.path.dirname(fs.projector_file))
    algorithm.save_projector(fs.projector_file)
    

    
def isv_estep(algorithm, iteration, indices, force=False):
  """Performs a single E-step of the ISV U matric training algorithm (parallel)"""

  fs = FileSelector.instance()
  stats_file = fs.isv_stats_file(iteration, indices[0], indices[1])

  if utils.check_file(stats_file, force, 1000):
    logger.info("ISV training: Skipping ISV E-Step since the file '%s' already exists", stats_file)
  else:
    logger.info("ISV training: E-Step from range(%d, %d)", *indices)

    # Temporary machine used for initialization
    algorithm.load_ubm(fs.ubm_file)

    # get the IVectorTrainer and call the initialization procedure
    trainer = algorithm.isv_trainer

    # Load data
    training_list = fs.training_list('projected_gmm', 'train_projector', arrange_by_client=True)
    data = [algorithm.read_gmm_stats(training_list[i]) for i in range(indices[0], indices[1])]
    data_initialize = [algorithm.read_gmm_stats(training_list[i]) for i in range(0,len(training_list))]

    # Load machine
    if iteration:
      # load last ISV file
      isv_base     = bob.learn.em.ISVBase(bob.io.base.HDF5File(fs.isv_intermediate_file(iteration)))
      isv_base.ubm = algorithm.ubm
    else:
      # create new ISV Base
      isv_base = bob.learn.em.ISVBase(algorithm.ubm, algorithm.subspace_dimension_of_u)

    # Perform the E-step     
    trainer.initialize(isv_base, data_initialize, rng = algorithm.rng) #Just to reset the accumulators
    trainer.e_step(isv_base, data)

    # write results to file
    bob.io.base.create_directories_safe(os.path.dirname(stats_file))
    hdf5 = bob.io.base.HDF5File(stats_file, 'w')
    hdf5.set('acc_u_a1', trainer.acc_u_a1)
    hdf5.set('acc_u_a2', trainer.acc_u_a2)
    logger.info("ISV training: Wrote Stats file '%s'", stats_file)
    
    
    
def _read_stats(filename):
  """Reads accumulated ISV statistics from file"""
  logger.debug("ISV training: Reading stats file '%s'", filename)
  hdf5 = bob.io.base.HDF5File(filename)
  acc_u_a1  = hdf5.read('acc_u_a1')
  acc_u_a2  = hdf5.read('acc_u_a2')
  return acc_u_a1,acc_u_a2


def _accumulate(filenames):
  acc_u_a1, acc_u_a2 = _read_stats(filenames[0])
  for filename in filenames[1:]:
    acc_u_a1, acc_u_a2 = _read_stats(filename)
    acc_u_a1 += acc_u_a1
    acc_u_a2 += acc_u_a2

  return acc_u_a1, acc_u_a2
  
  
  
def isv_mstep(algorithm, iteration, number_of_parallel_jobs, force=False, clean=False):
  """Performs a single M-step of the ISV algorithm (non-parallel)"""
  fs = FileSelector.instance()

  old_machine_file = fs.isv_intermediate_file(iteration)
  new_machine_file = fs.isv_intermediate_file(iteration + 1)

  if  utils.check_file(new_machine_file, force, 1000):
    logger.info("ISV training: Skipping ISV M-Step since the file '%s' already exists", new_machine_file)
  else:
    # get the files from e-step
    training_list = fs.training_list('projected_gmm', 'train_projector', arrange_by_client=True)
    # try if there is one file containing all data
    if os.path.exists(fs.isv_stats_file(iteration, 0, len(training_list))):
      # load stats file
      statistics = _read_stats(fs.isv_stats_file(iteration, 0, len(training_list)))
    else:
      # load several files
      stats_files = []
      for job in range(number_of_parallel_jobs):
        job_indices = tools.indices(training_list, number_of_parallel_jobs, job+1)
        if job_indices[-1] >= job_indices[0]:
          stats_files.append(fs.isv_stats_file(iteration, job_indices[0], job_indices[-1]))
      # read all stats files
      statistics = _accumulate(stats_files)

    # Load machine
    algorithm.load_ubm(fs.ubm_file)
    if iteration:
      isv_base     = bob.learn.em.ISVBase(bob.io.base.HDF5File(old_machine_file))
      isv_base.ubm = algorithm.ubm
    else:
      isv_base = bob.learn.em.ISVBase(algorithm.ubm, algorithm.subspace_dimension_of_u)

    # Creates the IVectorTrainer and initialize values
    trainer = algorithm.isv_trainer
    data_initialize = [algorithm.read_gmm_stats(training_list[i]) for i in range(0,len(training_list))]
    trainer.initialize(isv_base, data_initialize) #Just to allocate memory
    trainer.acc_u_a1 = statistics[0]
    trainer.acc_u_a2 = statistics[1]
    trainer.m_step(isv_base) # data is not used in M-step
    logger.info("ISV training: Performed M step %d", iteration)

    # Save the ISV model
    bob.io.base.create_directories_safe(os.path.dirname(new_machine_file))
    isv_base.save(bob.io.base.HDF5File(new_machine_file, 'w'))
    logger.info("ISV training: Wrote new ISV Base '%s'", new_machine_file)

  if iteration == algorithm.isv_training_iterations-1:
    shutil.copy(new_machine_file, fs.isv_file)
    logger.info("ISV training: Wrote new TV matrix '%s'", fs.isv_file)

  if clean and iteration > 0:
    old_dir = os.path.dirname(fs.isv_intermediate_file(iteration-1))
    logger.info("Removing old intermediate directory '%s'", old_dir)
    shutil.rmtree(old_dir)
  

def save_isv_projector(algorithm, force=False):
  fs = FileSelector.instance()
  if utils.check_file(fs.projector_file, force, 1000):
    logger.info("- Projector '%s' already exists.", fs.projector_file)
  else:
    # save the projector into one file
    algorithm.load_ubm(fs.ubm_file)
    algorithm.load_isv(fs.isv_file)
    logger.info("Writing projector into file %s", fs.projector_file)
    algorithm.save_projector(fs.projector_file)

    
