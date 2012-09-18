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

import sys, os
import scipy.io
import numpy as np

import msmbuilder.io
from msmbuilder import MSMLib
from msmbuilder import lumping
from msmbuilder import arglib
import logging
logger = logging.getLogger(__name__)

float_or_none = lambda s: None if s.lower() == 'none' else float(s)

def run_pcca(num_macrostates, assignments, tProb, output_dir):
    MacroAssignmentsFn = os.path.join(output_dir, "MacroAssignments.h5")
    MacroMapFn = os.path.join(output_dir, "MacroMapping.dat")
    arglib.die_if_path_exists([MacroAssignmentsFn, MacroMapFn])

    logger.info("Running PCCA...")
    MAP = lumping.PCCA(tProb, num_macrostates)

    # MAP the new assignments and save, make sure don't
    # mess up negaitve one's (ie where don't have data)
    MSMLib.apply_mapping_to_assignments(assignments, MAP)

    np.savetxt(MacroMapFn, MAP, "%d")
    msmbuilder.io.saveh(MacroAssignmentsFn, assignments)
    
    logger.info("Saved output to: %s, %s", MacroAssignmentsFn, MacroMapFn)
    
def run_pcca_plus(num_macrostates, assignments, tProb, output_dir, flux_cutoff=0.0,objective_function="crispness",do_minimization=True):
    MacroAssignmentsFn = os.path.join(output_dir, "MacroAssignments.h5")
    MacroMapFn = os.path.join(output_dir, "MacroMapping.dat")
    ChiFn = os.path.join(output_dir, 'Chi.dat')
    AFn = os.path.join(output_dir, 'A.dat')
    arglib.die_if_path_exists([MacroAssignmentsFn, MacroMapFn, ChiFn, AFn])
    
    logger.info("Running PCCA+...")
    A, chi, vr, MAP = lumping.pcca_plus(tProb, num_macrostates, flux_cutoff=flux_cutoff,
        do_minimization=do_minimization, objective_function=objective_function)

    MSMLib.apply_mapping_to_assignments(assignments, MAP)    

    np.savetxt(ChiFn, chi)
    np.savetxt(AFn, A)
    np.savetxt(MacroMapFn, MAP,"%d")
    msmbuilder.io.saveh(MacroAssignmentsFn, assignments)
    logger.info('Saved output to: %s, %s, %s, %s', ChiFn, AFn, MacroMapFn, MacroAssignmentsFn)


if __name__ == "__main__":
    parser = arglib.ArgumentParser(description="""
Applies the (f)PCCA(+) algorithm to lump your microstates into macrostates. You may
specify a transition matrix if you wish - this matrix is used to determine the
dynamics of the microstate model for lumping into kinetically relevant
macrostates.

Output: MacroAssignments.h5, a new assignments HDF file, for the Macro MSM.""")
    parser.add_argument('assignments', default='Data/Assignments.Fixed.h5')
    parser.add_argument('num_macrostates', type=int)
    parser.add_argument('tProb')
    parser.add_argument('output_dir')
    parser.add_argument('algorithm', help='Which algorithm to use', choices=['PCCA', 'PCCA+'], default='PCCA')
    parser.add_argument_group('Extra PCCA+ Options')
    parser.add_argument('flux_cutoff', help='''Discard eigenvectors below
        this flux''', default='None', type=float_or_none)
    parser.add_argument('objective_function', help='''Minimize which PCCA+
        objective function (crisp_metastability, metastability, or crispness)''',
                        default="crisp_metastability")
    parser.add_argument('do_minimization', help='Use PCCA+ minimization', default=True)
    args = parser.parse_args()

    # load args
    try:
        assignments = msmbuilder.io.loadh(args.assignments, 'arr_0')
    except KeyError:
        assignments = msmbuilder.io.loadh(args.assignments, 'Data')

    tProb = scipy.io.mmread(args.tProb)
    
    
    if args.do_minimization in ["False", "0"]:#workaround for arglib funniness?
        args.do_minimization = False
    else:
        args.do_minimization = True
    
    if args.algorithm == 'PCCA':
        run_pcca(args.num_macrostates, assignments, args.tProb, args.output_dir)
    elif args.algorithm == 'PCCA+':
        run_pcca_plus(args.num_macrostates, assignments, tProb, args.output_dir,
                      args.flux_cutoff, objective_function=args.objective_function,
                      do_minimization=args.do_minimization)
    else:
        raise Exception()
