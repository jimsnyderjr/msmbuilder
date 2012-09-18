# run tests on the project class
from msmbuilder import Project
import numpy.testing as npt
import os
from nose.tools import ok_, eq_

def test_project_1():
    'ensure that the counting of errors works right'
    records = {'conf_filename': None,
               'traj_lengths': [0,0,0],
               'traj_errors': [None, 1, None],
               'traj_paths': ['t0', 't1', 't2'],
               'traj_converted_from': [None, None, None]}
    proj = Project(records, validate=False)

    eq_(proj.n_trajs, 2)
    eq_(os.path.basename(proj.traj_filename(0)), 't0')

    # since t1 should be skipped
    eq_(os.path.basename(proj.traj_filename(1)), 't2')

@npt.raises(ValueError)
def test_project_2():
    'inconsistent lengths should be detected'
    records = {'conf_filename': None,
               'traj_lengths': [0,0], # this is one too short
               'traj_errors': [None, None, None],
               'traj_paths': ['t0', 't1', 't2'],
               'traj_converted_from': [None, None, None]}
    proj = Project(records, validate=False)
