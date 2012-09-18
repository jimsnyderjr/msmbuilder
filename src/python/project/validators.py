import abc
import numpy as np
from msmbuilder import Trajectory


class ValidationError:
    "All validation errors should subclass me"
    pass


class ExplosionError(ValidationError):
    "I get thrown by validators that check for explosion"
    pass

# All of the validators must be callables. they should raise an ValidationError
# when they fail, or else return None


class ExplosionValidator(object):
    def __init__(self, structure_or_filename, metric, max_distance):

        if isinstance(structure_or_filename, Trajectory):
            conf = structure_or_filename
        elif isinstance(structure_or_filename, basestring):
            conf = Trajectory.LoadTrajectoryFile(structure_or_filename)

        self.max_distance = max_distance
        self.metric = metric
        self._pconf = self.metric.prepare_trajectory(conf)

    def __call__(self, traj):
        ptraj = self.metric.prepare_trajectory(traj)
        distances = self.metric.one_to_all(self._pconf, ptraj)
        if np.any(distances > self.max_distance):
            i = np.where(distances > self.max_distance)[0][0]  # just get the first
            d = distances[i]
            raise ExplosionError('d(conf, frame[%d])=%f; greater than %s; metric=%s' % (i, d, self.max_distance, self.metric))


class RMSDExplosionValidator(ExplosionValidator):
    def __init__(self, structure_or_filename, max_rmsd, atom_indices=None):
        from msmbuilder.metrics import RMSD
        metric = RMSD(atom_indices)
        super(RMSDExplosionValidator, self).__init__(structure_or_filename, metric, max_rmsd)
