# This file is part of MSMBuilder.
#
# Copyright 2011 Stanford University
#
# MSMBuilder is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

import os
import numpy as np
import yaml
from msmbuilder.trajectory import Trajectory
from msmbuilder import msmio
import logging
from msmbuilder.utils import keynat
logger = logging.getLogger('project')

class Project(object):
    
    # this maps the private instance variable names to
    # the names that they can be serialized under
    mapping = {
        # the default name comes first, and alternate names
        # can come after
        '_conf_filename': 'conf_filename',
        '_traj_lengths': 'traj_lengths',
        '_traj_paths': 'traj_paths',
        '_traj_converted_from': 'traj_converted_from',
        '_traj_errors' : 'traj_errors',
    }

    @property
    def conf_filename(self):
        """Filename of the project topology (PDB)"""
        return os.path.normpath(os.path.join(self._project_dir, self._conf_filename))

    @property
    def n_trajs(self):
        """Number of trajectories in the project"""
        return len(self._traj_lengths)

    @property
    def traj_lengths(self):
        """Length of each of the trajectories, in frames"""
        return self._traj_lengths
    
    def __init__(self, records, validate=True, project_dir='.'):
        """Create a project from a  set of records
        
        Parameters
        ----------
        records : dict
            The data, either constructed or loaded from disk. If you
            provide insufficient data, we an error will be thrown.
        validate : bool, optional
            If true, some checks for consistency are done
        project_dir : string
            Base directory for the project. Filenames in the records
            dict are assumed to be given relative to this directory

        Notes
        -----
        This method is generally internally. To load projects from disk,
        use `Project.load_from`
            
        See Also
        --------
        load_from : to load from disk
        """
        
        self._conf_filename = None
        self._traj_lengths = None
        self._traj_paths = None
        self._traj_converted_from = None
        self._project_dir = os.path.abspath(project_dir)
                
        records_keys = set(records.keys())
        for key, val in self.mapping.iteritems():
            if val in records_keys:
                setattr(self, key, records[val])
        
        for key in self.mapping.keys():
            if getattr(self, key) is None:
                raise ValueError("attribute missing: %s" % key)
                
        if validate:
            self._validate()
        
    def __repr__(self):
        "Return a string representation of the project"
        return 'Project<(%d trajectories, from %s to %s  topology: %s)>' % (self.n_trajs, self.traj_filename(0), self.traj_filename(self.n_trajs-1), self.conf_filename)
    
    @classmethod
    def load_from(cls, filename):
        """
        Load project from disk
        
        Parameters
        ----------
        filename : string
            filename_or_file can be a path to a legacy .h5 or current
            .yaml file.
        
        Returns
        -------
        project : the loaded project object
        
        """

        rootdir = os.path.abspath(os.path.dirname(filename))
            
        if filename.endswith('.yaml'):
            with open(filename) as f:
                records = yaml.load(f)
                    
        elif filename.endswith('.h5'):
            records = msmio.loadh(filename, deferred=False)
                
            # UGLY: everything comes out of the .h5 file as an array
            # even the stuff like ConfFilename that is really just
            # a string
            for key, value in records.iteritems():
                if key in cls.mapping['_traj_lengths']:
                    records[key] = value.tolist()
                else:
                    records[key] = np.asscalar(value)
        else:
            raise ValueError('Sorry, I can only open files in .yaml'
                             ' or .h5 format: %s' % filename)
                  
        return cls(records, validate=True, project_dir=rootdir)
    
    
    def save(self, filename_or_file):
        if isinstance(filename_or_file, basestring):
            if not filename_or_file.endswith('.yaml'):
                filename_or_file += '.yaml'
            dirname = os.path.abspath(os.path.dirname(filename_or_file))
            if not os.path.exists(dirname):
                logger.info("Creating directory: %s" % dirname)
                os.makedirs(dirname)
            handle = open(filename_or_file, 'w')
            own_fid = True
        elif isinstance(filename_or_file, file):
            dirname = os.path.abspath(os.path.dirname(filename_or_file.name))
            handle = filename_or_file
            own_fid = False
        
        # somewhat complicated logic if the directory you're
        # saving in is different than the directory this
        # project references its paths from

        # the point is that the when the file lists paths, those
        # paths are going to be interpreted as being with respect to
        # the directory that the file is in. So when the Project file
        # is being resaved (but the Trajectorys are not being moved)
        # then the paths need to change to compensate
        
        relative = os.path.relpath(self._project_dir, os.path.dirname(filename_or_file))

        records = {'trajs': []}
        records['conf_filename'] = os.path.join(relative, self._conf_filename)
        traj_paths = [os.path.join(relative, path) for path in self._traj_paths]
        for i in xrange(len(traj_paths)):
            records['trajs'].append({'id': i,
                                    'path': traj_paths[i],
                                    'converted_from': self._traj_converted_from[i],
                                    'length': self._traj_lengths[i],
                                    'errors': self._traj_errors[i]})
        
        yaml.dump(records, handle)
        
        if own_fid:
            handle.close()
            
        return filename_or_file

    def load_traj(self, trj_index, stride=1):
        filename = self.traj_filename(trj_index)
        return Trajectory.LoadTrajectoryFile(filename, Stride=stride)

    def load_conf(self):
        return Trajectory.LoadTrajectoryFile(self.conf_filename)

    def traj_filename(self, traj_index):
        return os.path.normpath(os.path.join(self._project_dir, self._traj_paths[traj_index]))

    def _validate(self):
        if not os.path.exists(self.conf_filename):
            raise ValueError('conf does not exist: %s' % self.conf_filename)
        for i in xrange(len(self._traj_paths)):
            if not os.path.exists(self.traj_filename(i)):
                raise ValueError("%s does not exist" % self.traj_filename(i))
        if not np.all(self.traj_lengths == self._eval_traj_lengths()):
            raise ValueError('Trajs length don\'t match what\'s on disk')

    
    def empty_traj(self):
        traj = self.load_conf()
        traj['XYZList'] = None
        return traj
        
    def _eval_traj_lengths(self, as_numpy=False):
        traj_lengths = np.zeros(self.n_trajs)
        for i in xrange(self.n_trajs):
            traj_lengths[i] = Trajectory.LoadTrajectoryFile(self.traj_filename(i), JustInspect=True, Conf=self.conf_filename)[0]
        return traj_lengths


def sanitize_type(obj):
    clean_types = [str, int, float, list, dict]
    if any(isinstance(obj, e) for e in clean_types):
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise Exception('Cannot sanitize %s' % obj)