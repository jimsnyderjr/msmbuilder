#!/usr/bin/env python
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
import sys
import numpy

from msmbuilder import CreateMergedTrajectoriesFromFAH
from msmbuilder import Serializer
from msmbuilder import Project
from msmbuilder import arglib
import logging
logger = logging.getLogger(__name__)


def run(atomindicesFn, pdbFn, trajlistFn, datatype):
    logger.warning("This script assumes you have already removed duplicate frames from your input trajectories if they exist.")

    # check directory structure
    if os.path.exists("Data/Assignments.h5"):
        logger.error("Data/Assignments.h5 already exists.  If you really want to recreate it, then delete Data/Assignments.h5 and re-run this script.")
        sys.exit(0)
    if os.path.exists("ProjectInfo.h5"):
        logger.error("ProjectInfo.h5 already exists.  If you really want to recreate it, then delete ProjectInfo.h5 and re-run this script.")
        sys.exit(0)
    if os.path.exists("Trajectories/trj0.%s" % datatype):
        logger.warning("The trajectories seem to have been converted to the desired format already.  This script will now assume the contents of the Trajectories directory are valid  If this is not the case, then you should delete or rename your Trajectories directory and re-run this script.")
    # don't make Trajectories directory because done later in CreateMergedTrajectories method call
    if not os.path.exists("Data"):
        os.mkdir("Data")

    # get list of trajectories
    trajList = []
    f = open(trajlistFn, 'r')
    for line in f:
        traj = line.strip()
        trajList.append(traj)
    f.close()

    # find maximum length of any trajectory
    logger.info("Converting assignments to hdf5.")
    maxLen = 0
    nMicro = 0
    LenList = []
    i = 0
    while i < len(trajList):
        traj = trajList[i]
        assignFn = os.path.join("assignments", traj)
        try:
            a = numpy.loadtxt(assignFn, dtype=int)
        except:
            trajList.remove(traj)
            continue
        aLen = len(a)
        if aLen > maxLen:
            maxLen = aLen
            maxStateInd = a.max()
        if maxStateInd > nMicro:
            nMicro = maxStateInd
        LenList.append(aLen)
        i += 1
    nMicro += 1 # account for fact numbering of states starts at 0
    nFiles = len(trajList)

    assigns = -1*numpy.ones((nFiles, maxLen), 'int32')
    i = 0
    for traj in trajList:
        assignFn = os.path.join("assignments", traj)
        a = numpy.loadtxt(assignFn, dtype=int)
  
        # store just first column (microstate assignments)
        if len(a.shape) == 1:
            assigns[i,0:a.shape[0]] = a[:]
        else:
            assigns[i,0:a.shape[0]] = a[:,0]
        i += 1

    # create assignments hdf5 file
    Serializer.save_data("Data/Assignments.h5",assigns)

    # load atom indices if present
    if atomindicesFn != None:
        try:
            atomInd = numpy.loadtxt(atomindicesFn, dtype=int)
        except:
            atomInd = None
    else:
        atomInd = None

    # setup trajectories
    firstTrajFile = os.path.join("trajectories", trajList[0])
    logger.info(firstTrajFile)
    if not os.path.exists(firstTrajFile ):
        logger.warning("No trajectory data found, no trajectory conversions to be done.")

        logger.info("Creating Project File based on assignments")
        DictContainer = {"TrajLengths": numpy.array(LenList),
                        "TrajFilePath": "Trajectories",
                        "TrajFileBaseName": "trj",
                        "TrajFileType": ".lh5",
                        "ConfFilename": pdbFn}
        P1 = Project(DictContainer)
        P1.save_to_hdf("ProjectInfo.h5")
    else:
        logger.warning("Found trajectory data. Moving old trajectories directory to one called old_trajectories and converting to %s files in new Trajectories directory.", datatype)
        os.system("mv trajectories old_trajectories")
        listOfXtcLists = []
        for traj in trajList:
            trajPartListFn = os.path.join("old_trajectories", traj)
            trajPartList = []
            f = open(trajPartListFn, 'r')
            for line in f:
                trajPartName = line.strip()
                trajPartList.append("old_"+trajPartName+".xtc")
            f.close()
            listOfXtcLists.append(trajPartList)

        logger.info('xtcs: %s', listOfXtcLists)
        CreateMergedTrajectoriesFromFAH.CreateMergedTrajectories(pdbFn, listOfXtcLists, AtomIndices=atomInd)

        logger.info("Creating Project File")
        P1=Project.CreateProjectFromDir(ConfFilename=pdbFn,TrajFileType="."+datatype)

if __name__ == "__main__":
    parser = arglib.ArgumentParser(description="""
    Convert msmbuilder v0.1 and v1.0.1 style assignments to the v2 and hdf5 format.
    Also creates a ProjectInfo.h5 file.

    Notes:
    If you input atomindices to subselect atoms, you will need to create a new AtomIndices file for
    the clustering stages.  This because the atom numbers will change upon sub-selection.""")
    parser.add_argument('atom_indices', 'Path to atom indices file for RMSD')
    parser.add_argument('pdb')
    parser.add_argument('trajlist', help='Path to MSMBuilder1-style trajlist')
    parser.add_argument('datatype', help='Format to store data in.',
        choices=['lh5', 'h5', 'xtc'], default='lh5')
    args = parser.parse_args()

    run(args.atom_indices, args.pdb, args.trajlist, args.datatype)
  
