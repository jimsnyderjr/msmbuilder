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

"""
New tests for the scripts. The idea is to replace the current TestWrappers.py
with this. The improvements are basically towards maintainability, and are
documented in PR 48 at https://github.com/SimTk/msmbuilder/pull/48
"""

import numpy as np
import tempfile
import shutil
import os
import tarfile
from msmbuilder.testing import *

from msmbuilder.scripts import ConvertDataToHDF
from msmbuilder.scripts import CreateAtomIndices
from msmbuilder.scripts import Cluster
from msmbuilder.scripts import Assign
from msmbuilder.scripts import AssignHierarchical
from msmbuilder.scripts import BuildMSM
from msmbuilder.scripts import CalculateImpliedTimescales


def pjoin(*args):
    return os.path.join(*args)

# subclassing this will get you a test where you have a temporary 
# directory (self.td) that will get cleaned up when you tes finishes
# (whether it passes or fails)
class WTempdir(object):
    def setup(self):
        self.td = tempfile.mkdtemp()
    def teardown(self):
        shutil.rmtree(self.td)


class test_ConvertDataToHDF(WTempdir):
    def test(self):
        # extract xtcs to a temp dir
        xtc_fn = get('XTC.tgz', just_filename=True)
        with tarfile.open(xtc_fn, mode='r:gz') as fh:
            fh.extractall(self.td)
    
        outfn = pjoin(self.td, 'ProjectInfo.h5')
        # mode to that directory
        os.chdir(self.td)
        ConvertDataToHDF.run(projectfn=outfn,
                             PDBfn=get('native.pdb', just_filename=True),
                             InputDir=pjoin(self.td, 'XTC'),
                             source='file',
                             mingen=0,
                             stride=1,
                             rmsd_cutoff=np.inf,
                             parallel='None')

        eq(load(outfn), get('ProjectInfo.h5'))


def test_CreateAtomIndices():
    indices = CreateAtomIndices.run(get('native.pdb', just_filename=True),
                                   'minimal')    
    eq(indices, get('AtomIndices.dat'))


class test_Cluster_kcenters(WTempdir):
    # this one tests kcenters
    def test(self):
        args, metric = Cluster.parser.parse_args([
            '-p', get('ProjectInfo.h5', just_filename=True),
            '-a', pjoin(self.td, 'Assignments.h5'),
            '-d', pjoin(self.td, 'Assignments.h5.distances'),
            '-g', pjoin(self.td, 'Gens.lh5'),
            'rmsd', '-a', get('AtomIndices.dat', just_filename=True), 
            'kcenters', '-k', '100'], print_banner=False)
        Cluster.main(args, metric)

        eq(load(pjoin(self.td, 'Assignments.h5')),
           get('Assignments.h5'))
        eq(load(pjoin(self.td, 'Assignments.h5.distances')),
           get('Assignments.h5.distances'))
        eq(load(pjoin(self.td, 'Gens.lh5')),
           get('Gens.lh5'))


class test_Cluster_hierarchical(WTempdir):
   # this one tests hierarchical
   def test(self):
       args, metric = Cluster.parser.parse_args([
           '-p', get('ProjectInfo.h5', just_filename=True),
           '-s', '10',
           '-g', pjoin(self.td, 'Gens.lh5'),
           'rmsd', '-a', get('AtomIndices.dat', just_filename=True), 
           'hierarchical', '-o', pjoin(self.td, 'ZMatrix.h5')],
            print_banner=False)
       Cluster.main(args, metric)

       eq(load(pjoin(self.td, 'ZMatrix.h5')),
            get('ZMatrix.h5'))


class test_Assign(WTempdir):
    def test(self):
        args, metric = Assign.parser.parse_args([
            '-p', get('ProjectInfo.h5', just_filename=True),
            '-g', get('Gens.lh5', just_filename=True),
            '-o', self.td,
            'rmsd', '-a', get('AtomIndices.dat', just_filename=True)], 
            print_banner=False)
        Assign.main(args, metric)
        
        eq(load(pjoin(self.td, 'Assignments.h5')),
           get('Assignments.h5'))
        eq(load(pjoin(self.td, 'Assignments.h5.distances')),
           get('Assignments.h5.distances'))


def test_AssignHierarchical():
    asgn = AssignHierarchical.main(k=100, d=None,
        zmatrix_fn=get('ZMatrix.h5', just_filename=True))
    
    eq(asgn, get('WardAssignments.h5')['Data'])


class test_BuildMSM(WTempdir):
    def test(self):
        BuildMSM.run(LagTime=1, assignments=get('Assignments.h5')['Data'], Symmetrize='MLE',
            OutDir=self.td)
        
        eq(load(pjoin(self.td, 'tProb.mtx')), get('tProb.mtx'))
        eq(load(pjoin(self.td, 'tCounts.mtx')), get('tCounts.mtx'))
        eq(load(pjoin(self.td, 'Mapping.dat')), get('Mapping.dat'))
        eq(load(pjoin(self.td, 'Assignments.Fixed.h5')), get('Assignments.Fixed.h5'))
        eq(load(pjoin(self.td, 'Populations.dat')), get('Populations.dat'))
        
def test_CalculateImpliedTimescales():
    impTimes = CalculateImpliedTimescales.run(MinLagtime=3, MaxLagtime=5,
        Interval=1, NumEigen=10, AssignmentsFn=get('Assignments.h5', just_filename=True),
        symmetrize='Transpose', nProc=1)
    
    eq(impTimes, get('ImpliedTimescales.dat'))




        