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
import msmbuilder.io
from msmbuilder.clustering import Hierarchical
from msmbuilder import arglib
import logging
logger = logging.getLogger(__name__)

def main():
    parser = arglib.ArgumentParser(description='Assign data using a hierarchical clustering')
    parser.add_argument('hierarchical_clustering_zmatrix', default='./Data/Zmatrix.h5', 
        help='Path to hierarchical clustering zmatrix' )
    parser.add_argument('num_states', help='Number of States', default='none')
    parser.add_argument('cutoff_distance', help='Maximum cophenetic distance', default='none')
    parser.add_argument('assignments', type=str)
    args = parser.parse_args()
    
    k = int(args.num_states) if args.num_states != 'none' else None
    d = float(args.cutoff_distance) if args.cutoff_distance != 'none' else None
    if k is None and d is None:
        logger.error('You need to supply either a number of states or a cutoff distance')
        sys.exit(1)
    
    arglib.die_if_path_exists(args.assignments)
    
    assignments = hierarchical_clustering_zmatrix.get_assignments(k=k, cutoff_distance=d)
    
    msmbuilder.io.saveh(args.assignments, assignments)
    logger.info('Saved assignments to %s', args.assignments)
    
if __name__ == "__main__":
    main()

