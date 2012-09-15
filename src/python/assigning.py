import sys, os
import numpy as np
import tables
from msmbuilder.Trajectory import Trajectory
import logging
logger = logging.getLogger('assigning')

def _setup_containers(outputdir, project):
    """
    Setup the files on disk (Assignments.h5 and Assignments.h5.distances) that
    results will be sent to.
    
    Check to ensure that if they exist (and contain partial results), the
    containers are not corrupted
    
    Parameters
    ----------
    outputdir : str
        path to save/find the files
    project : msmbuilder.Project
        The msmbuilder project file. Only the NumTrajs and TrajLengths are
        actully used (if you want to spoof it, you can just pass a dict)
    all_vtrajs : list
        The VTrajs are used to check that the containers on disk, if they
        exist, contain the right stuff
        
    Returns
    -------
    f_assignments : tables.File
        pytables handle to the assignments file, open in 'append' mode
    f_distances : tables.File
        pytables handle to the assignments file, open in 'append' mode    
    """
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    
    assignments_fn = os.path.join(outputdir, 'Assignments.h5')
    distances_fn = os.path.join(outputdir, 'Assignments.h5.distances')
    
    max_n_frames = np.max(project.traj_lengths)
    minus_ones = -1 * np.ones((n_trajs, max_n_frames))
    
    def save_container(filename, dtype):
        msmio.saveh(filename, Data=np.array(minus_ones, dtype=dtype),
                    completed_trajs=np.zeros((project.n_trajs), dtype=np.bool))
    
    def check_container(filename):
        ondisk = msmio.load(filename, deferred=False)
        if project.n_trajs != len(ondisk['hashes']):
            raise ValueError('You jabe %d ntrajs, but your checkpoint \
file has %d' % (project.n_trajs, len(ondisk['completed_trajs'])))
    
    # save assignments container
    if (not os.path.exists(assignments_fn)) \
            and (not os.path.exists(distances_fn)):
        save_container(assignments_fn, np.int)
        save_container(distances_fn, np.float32)
    elif os.path.exists(assignments_fn) and os.path.exists(distances_fn):
        check_container(assignments_fn)
        check_container(distances_fn)
    else:
        raise ValueError("You're missing one of the containers")
    
    # append mode is read and write
    f_assignments = tables.openFile(assignments_fn, mode='a')
    f_distances = tables.openFile(distances_fn, mode='a')
    
    return f_assignments, f_distances
def assign_in_memory(metric, generators, project):
    """This really should be called simple assign -- its the simplest"""

    n_trajs, max_traj_length = project['NumTrajs'], max(project['TrajLengths'])
    assignments = -1 * np.ones((n_trajs, max_traj_length), dtype='int')
    distances = -1 * np.ones((n_trajs, max_traj_length), dtype='float32')

    pgens = metric.prepare_trajectory(generators)
    
    for i in xrange(n_trajs):
        traj = project.LoadTraj(i)
        ptraj = metric.prepare_trajectory(traj)
        for j in xrange(len(traj)):
            d = metric.one_to_all(ptraj, pgens, j)
            assignments[i,j] = np.argmin(d)
            distances[i,j] = d[assignments[i,j]]

    return assignments, distances
    

def assign_with_checkpoint(metric, project, generators, assignments_path, distances_path, checkpoint=1):
    """Assign each of the frames in each of the trajectories in the supplied project to
    their closest generator (frames of the trajectory "generators") using the supplied
    distance metric.
    
    assignments_path and distances_path should be the filenames of to h5 files in which the
    results are/will be stored. The results are stored along the way as they are computed,
    and if this method is killed halfway through execution, you can restart it and it
    will not have lost its place (i.e. checkpointing)"""
    pgens = metric.prepare_trajectory(generators)
    
    num_gens = len(pgens)
    num_trajs = project['NumTrajs']
    longest = max(project['TrajLengths'])
    
    assignments_tmp, distances_tmp, all_assignments, all_distances, completed_trajectories = _setup_containers(assignments_path,
        distances_path, num_trajs, longest)
    
    for i in xrange(project['NumTrajs']):
        if completed_trajectories[i]:
            logger.info('Skipping trajectory %s -- already assigned', i)
            continue
        logger.info('Assigning trajectory %s', i)
        
        ptraj = metric.prepare_trajectory(project.LoadTraj(i))
        distances = -1 * np.ones(len(ptraj), dtype=np.float32)
        assignments = -1 * np.ones(len(ptraj), dtype=np.int)
        
        for j in range(len(ptraj)):
            d = metric.one_to_all(ptraj, pgens, j)
            ind = np.argmin(d)
            assignments[j] = ind
            distances[j] = d[ind]
            
        all_distances[i, 0:len(ptraj)] = distances
        all_assignments[i, 0:len(ptraj)] = assignments
        completed_trajectories[i] = True
        # checkpoint every few trajectories
        if ((i+1) % checkpoint == 0) or (i+1 == project['NumTrajs']):
            Serializer({'Data': all_assignments,
                        'completed_trajectories': completed_trajectories
                        }).SaveToHDF(assignments_tmp)
            Serializer({'Data': all_distances}).SaveToHDF(distances_tmp)
            os.rename(assignments_tmp, assignments_path)
            os.rename(distances_tmp, distances_path)

    return all_assignments, all_distances

def streaming_assign_with_checkpoint(metric, project, generators, assignments_path, distances_path, checkpoint=1,chunk_size=10000):
    """Assign each of the frames in each of the trajectories in the supplied project to
    their closest generator (frames of the trajectory "generators") using the supplied
    distance metric.
    
    assignments_path and distances_path should be the filenames of to h5 files in which the
    results are/will be stored. The results are stored along the way as they are computed,
    and if this method is killed halfway through execution, you can restart it and it
    will not have lost its place (i.e. checkpointing)

    This version is the same as the original assign_with_checkpoint function, except that 
    it streams through each trajectory rather than loading the entire trajectory at a time"""

    if not project['TrajFileType'] in ['.lh5', '.h5']:
        warnings.warn("Streaming assign currently only works with .lh5 or .h5 files.")
        return assign_with_checkpoint( metric, project, generators, assignments_path, distances_path, checkpoint )

    pgens = metric.prepare_trajectory(generators)
    
    num_gens = len(pgens)
    num_trajs = project['NumTrajs']
    longest = max(project['TrajLengths'])
    
    assignments_tmp, distances_tmp, all_assignments, all_distances, completed_trajectories = _setup_containers(assignments_path,
        distances_path, num_trajs, longest)
    
    for i in xrange(project['NumTrajs']):
        if completed_trajectories[i]:
            print 'Skipping trajectory %d -- already assigned' % i
            continue
        print 'Assigning trajectory %d' % i
 
        distances = []
        assignments = []       

        for chunk_trj in Trajectory.EnumChunksFromLHDF( project.GetTrajFilename(i), ChunkSize=chunk_size ):

            print "Chunked."
            chunk_ptraj = metric.prepare_trajectory( chunk_trj )
            chunk_distances = -1 * np.ones(len(chunk_ptraj), dtype=np.float32)
            chunk_assignments = -1 * np.ones(len(chunk_ptraj), dtype=np.int)

            for j in range(len(chunk_ptraj)):
                d = metric.one_to_all(chunk_ptraj, pgens, j)
                ind = np.argmin(d)
                chunk_assignments[j] = ind
                chunk_distances[j] = d[ind]
            distances.extend(chunk_distances)
            assignments.extend(chunk_assignments)
            
        all_distances[i, 0:len(distances)] = np.array( distances )
        all_assignments[i, 0:len(distances)] = np.array( assignments )
        completed_trajectories[i] = True
        # checkpoint every few trajectories
        if ((i+1) % checkpoint == 0) or ((i+1) == project['NumTrajs']):
            Serializer({'Data': all_assignments,
                        'completed_trajectories': completed_trajectories
                        }).SaveToHDF(assignments_tmp)
            Serializer({'Data': all_distances}).SaveToHDF(distances_tmp)
            os.rename(assignments_tmp, assignments_path)
            os.rename(distances_tmp, distances_path)

    return all_assignments, all_distances

    

