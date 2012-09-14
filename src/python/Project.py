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
import IPython as ip
import msmio

class Project(object):
    mapping = {
        '_conf_filename': ['conf_filename', 'ConfFilename'],
        '_n_trajs': ['n_trajs', 'NumTrajs'],
        '_traj_lengths': ['TrajLengths', 'traj_lengths'],
        '_traj_basename': ['traj_basename', 'TrajFileBaseName'],
        '_traj_path': ['traj_path', 'TrajFilePath'],
        '_traj_ext': ['traj_ext', 'TrajFileType']
    }
    
    @property
    def conf_filename(self):
        return os.path.join(self._project_dir, self._conf_filename)
    @property
    def n_trajs(self):
        return self._n_trajs
    @property
    def traj_lengths(self):
        return np.array(self._traj_lengths)
    @property
    def traj_basename(self):
        return self._traj_basename
    @property
    def traj_path(self):
        return os.path.normpath(os.path.join(self._project_dir, self._traj_path))
    @property
    def traj_ext(self):
        return self._traj_ext
    
    def __init__(self, records, validate=True, project_dir='.'):
        self._conf_filename = None
        self._n_trajs = None
        self._traj_lengths = None
        self._traj_basename = None
        self._traj_ext = None
        self._conf_filename = None
        
        records_keys = set(records.keys())
        for key, val in self.mapping.iteritems():
            match = list(set(val) & records_keys)
            if match:
                assert len(match) == 1, 'duplicate keys detected'
                setattr(self, key, records[match[0]])
        
        for key in self.mapping.keys():
            if getattr(self, key) is None:
                raise ValueError("attribute missing: %s" % key)
        
        self._project_dir = os.path.abspath(os.path.dirname(os.path.abspath(project_dir)))
        
        if validate:
            self._validate()
        
    def __repr__(self):
        return 'Project<(' + {'traj_path': self.traj_path,
                'traj_ext': self.traj_ext, 'traj_basename': self.traj_basename,
                'n_trajs': self.n_trajs, 'traj_lengths': self.traj_lengths,
                'conf_filename': self.conf_filename}.__repr__() + ")>"
    
    @classmethod
    def load_from(cls, filename_or_file):
        if isinstance(filename_or_file, basestring):
            rootdir = filename_or_file
            if filename_or_file.endswith('.yaml'):
                with open(filename_or_file) as f:
                    records = yaml.load(f)
            elif filename_or_file.endswith('.h5'):
                records = msmio.loadh(filename_or_file, deferred=False)
                
                # UGLY: everything comes out of the .h5 file as an array
                # even the stuff like ConfFilename that is really just
                # a string
                for key, value in records.iteritems():
                    if key in cls.mapping['_traj_lengths']:
                        records[key] = value.tolist()
                    else:
                        records[key] = np.asscalar(value)
                        

        elif isinstance(filename_or_file, file):
            rootdir = filename_or_file.name
            records = yaml.load(file)
            
        else:
            raise TypeError('type=%s is not supported' % filename_or_file)
        
        return cls(records, validate=True, project_dir=rootdir)
    
    def save(self, filename_or_file):
        if isinstance(filename_or_file, basestring):
            if not filename_or_file.endswith('.yaml'):
                filename_or_file += '.yaml'
            dirname = os.path.dirname(filename_or_file)
            handle = open(filename_or_file, 'w')
            own_fid = True
        elif isinstance(filename_or_file, file):
            dirname = os.path.dirname(filename_or_file.name)
            handle = filename_or_file
            own_fid = False
            
        records = {}
        for key, val in self.mapping.iteritems():
            records[val[0]] = getattr(self, key)
        # somewhat complicated logic if the directory you're
        # saving in is different than the directory this
        # project references its paths from
        
        # the point is that the when the file lists paths, those
        # paths are going to be interpreted as being with respect to
        # the directory that the file is in. So when the Project file
        # is being resaved (but the Trajectorys are not being moved)
        # then the paths need to change to compensate
        relative = os.path.relpath(self._project_dir, os.path.dirname(filename_or_file))
        records['traj_path'] = os.path.join(relative, records['traj_path'])
        records['conf_filename'] = os.path.join(relative, records['conf_filename'])

        yaml.dump(records, handle)
        
        if own_fid:
            handle.close()

    def load_traj(self, trj_index, stride=1):
        filename = self.traj_filename(trj_index)
        return Trajectory.LoadTrajectoryFile(filename, stride=stride)
            
    def traj_filename(self, traj_index):
        return os.path.join(self._project_dir, self.traj_path,
                self.traj_basename + str(traj_index) + self.traj_ext)
        
    def _validate(self):
        if not self._n_trajs == len(self.traj_lengths):
            raise ValueError('traj lengths mismatch')
        if not os.path.exists(self.conf_filename):
            raise ValueError('conf does not exist: %s' % self.conf_filename)
        for i in xrange(self.n_trajs):
            if not os.path.exists(self.traj_filename(i)):
                raise ValueError("%s does not exist" % self.traj_filename(i))
    
    def empty_trajectory(self):
        traj = Trajectory.LoadTrajectoryFile(self.conf_filename)
        traj['XYZList'] = None
        return traj
    
if __name__ == '__main__':
    p = Project.load_from('Tutorial/Project.yaml')
    p2 = Project.load_from('TestFiles/UnitTestReference/ProjectInfo.h5')
    p2.save('sdf.yaml')