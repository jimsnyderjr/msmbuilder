"""
As the trajectories are being converted from their native format to
MSMBuilder's format (lh5), all of the registeted validtors will be run
against each trajectory. Validators should be callables like
functions or classes with a __call__ method that check a
trajectory. They are free to modify a trajectory as well, since it is
passed by reference to the validator.

If a validator detects a problem with a trajectory, it should raise
a ValidationError -- that is, an error which subclasses
msmbuilder.project.validators.ValidationError. When the ProjectBuilder
detects a ValidationError, the error will be recorded in the project
file, but the execution will procdede as normal and the trajectory will
still be saved to disk. It will just be marked specially as "in error".

In the current Project implementation, trajectories that are "in error"
will be ignored -- when using project.load_traj(), only the "valid"
trajectories will be returned, and project.n_trajs will only count
the valid trajectories.
"""
import numpy as np
from msmbuilder import Trajectory


class ValidationError(Exception):
    "All validation errors should subclass me"
    pass


class ExplosionError(ValidationError):
    "I get thrown by validators that check for explosion"
    pass

# All of the validators must be callables. they should raise an ValidationError
# when they fail, or else return None


class ExplosionValidator(object):
    def __init__(self, structure_or_filename, metric, max_distance):
        """Create an explosion validator
    
        Checks the distance from every frame to a structure and
        watches for things that are too far away
        
        Parameters
        ----------
        structure_or_filename : {msmbuilder.Trajectory, str}
            The structure to measure distances to, either as a trajectory (the first
            frame is the only one that counts) or a path to a trajectory
            on disk that can be loaded
        metric : msmbuilder distance metric
            Metric by which you want to measure distance
        max_distance : float
            The threshold distance, above which a ValidationError
            will be thrown
        """

        if isinstance(structure_or_filename, Trajectory):
            conf = structure_or_filename
        elif isinstance(structure_or_filename, basestring):
            conf = Trajectory.LoadTrajectoryFile(structure_or_filename)

        self.max_distance = max_distance
        self.metric = metric
        self._pconf = self.metric.prepare_trajectory(conf)

    def __call__(self, traj):
        ptraj = self.metric.prepare_trajectory(traj)
        distances = self.metric.one_to_all(self._pconf, ptraj, 0)
        if np.any(distances > self.max_distance):
            i = np.where(distances > self.max_distance)[0][0]  # just get the first
            d = distances[i]
            raise ExplosionError('d(conf, frame[%d])=%f; greater than %s; metric=%s' % (i, d, self.max_distance, self.metric))

class RMSDExplosionValidator(ExplosionValidator):
    """Validator that checks for explosion by measuring the RMSD of every frame
    to a PDB and watching for values which are too high"""
    
    def __init__(self, structure_or_filename, max_rmsd, atom_indices=None):
        """Create an RMSD validator
        
        Parameters
        ----------
        structure_or_filename : {msmbuilder.Trajectory, str}
            The structure to measure distances to, either as a trajectory (the first
            frame is the only one that counts) or a path to a trajectory
            on disk that can be loaded
        max_rmsd : float
            The threshold rmsd
        atom_indices : np.array [ndim=1, dtype=int]
            The indices over which you want to measure RMSD
        """
        # delay the import of RMSD
        from msmbuilder.metrics import RMSD
        metric = RMSD(atom_indices)
        super(RMSDExplosionValidator, self).__init__(structure_or_filename, metric, max_rmsd)


class MinLengthValidator(object):
    
    
    def __init__(self, min_length):
        """
        A validator that discards trajectories with two little data. Useful for
        excluding trajectories that might be far too short to lend any kind of
        significance to an MSM.
        """
        self.min_length = min_length
        raise NotImplementedError()
        
    def __call__(self, traj):
        if len(traj) * traj.dt < self.min_length: # traj.dt here needs the timestep
            raise ExplosionError('trajectory shorter than requested cutoff: %f' % self.min_length)
        
        
        
        
