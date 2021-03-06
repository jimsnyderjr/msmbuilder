import sys
import os

import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.io

from msmbuilder import tpt
from msmbuilder import io
from common import reference_dir


class TestTPT():
    """ Test the transition_path_theory library """


    def setUp(self):
        
        # load in the reference data
        self.tpt_ref_dir = os.path.join(reference_dir(), "transition_path_theory_reference")
        self.tprob = scipy.io.mmread( os.path.join(self.tpt_ref_dir, "tProb.mtx") ) #.toarray()
        self.sources   = [0]   # chosen arbitarily by TJL
        self.sinks     = [70]  # chosen arbitarily by TJL
        self.waypoints = [60]  # chosen arbitarily by TJL
        self.lag_time  = 1.0   # chosen arbitarily by TJL

        # set up the reference data for hub scores
        self.hub_ref_dir = os.path.join(self.tpt_ref_dir, "hub_ref")
        K = np.loadtxt( os.path.join(self.hub_ref_dir, 'ratemat_1.dat') )
        #self.hub_T = scipy.linalg.expm( K ) # delta-t should not affect hub scores
        self.hub_T = np.transpose( np.genfromtxt(os.path.join(self.hub_ref_dir, 'mat_1.dat'))[:,:-3] )
        
        for i in range(self.hub_T.shape[0]):
            self.hub_T[i,:] /= np.sum(self.hub_T[i,:])
        
        self.hc = np.loadtxt( os.path.join(self.hub_ref_dir, 'fraction_visited.dat') )
        self.Hc = np.loadtxt( os.path.join(self.hub_ref_dir, 'hub_scores.dat') )[:,2]
        

    def test_committors(self):
        Q = tpt.calculate_committors(self.sources, self.sinks, self.tprob)
        Q_ref = io.loadh(os.path.join(self.tpt_ref_dir, "committors.h5"), 'Data')
        npt.assert_array_almost_equal(Q, Q_ref)
        
        
    def test_flux(self):
        
        flux = tpt.calculate_fluxes(self.sources, self.sinks, self.tprob)
        flux_ref = io.loadh(os.path.join(self.tpt_ref_dir,"flux.h5"), 'Data')
        npt.assert_array_almost_equal(flux.toarray(), flux_ref)
        
        net_flux = tpt.calculate_net_fluxes(self.sources, self.sinks, self.tprob)
        net_flux_ref = io.loadh(os.path.join(self.tpt_ref_dir,"net_flux.h5"), 'Data')
        npt.assert_array_almost_equal(net_flux.toarray(), net_flux_ref)
        
        
    def test_path_calculations(self):
        path_output = tpt.find_top_paths(self.sources, self.sinks, self.tprob)

        paths_ref = io.loadh(os.path.join(self.tpt_ref_dir,"dijkstra_paths.h5"), 'Data')
        fluxes_ref = io.loadh(os.path.join(self.tpt_ref_dir,"dijkstra_fluxes.h5"), 'Data')
        bottlenecks_ref = io.loadh(os.path.join(self.tpt_ref_dir,"dijkstra_bottlenecks.h5"), 'Data')

        #npt.assert_array_almost_equal(path_output[0], paths_ref)
        npt.assert_array_almost_equal(path_output[1], bottlenecks_ref)
        npt.assert_array_almost_equal(path_output[2], fluxes_ref)
        
        
    def test_mfpt(self):
        
        mfpt = tpt.calculate_mfpt(self.sinks, self.tprob, lag_time=self.lag_time)
        mfpt_ref = io.loadh(os.path.join(self.tpt_ref_dir, "mfpt.h5"), 'Data')
        npt.assert_array_almost_equal(mfpt, mfpt_ref)
        
        ensemble_mfpt = tpt.calculate_ensemble_mfpt(self.sources, self.sinks, self.tprob, self.lag_time)
        ensemble_mfpt_ref = io.loadh(os.path.join(self.tpt_ref_dir, "ensemble_mfpt.h5"), 'Data')
        npt.assert_array_almost_equal(ensemble_mfpt, ensemble_mfpt_ref)
        
        all_to_all_mfpt = tpt.calculate_all_to_all_mfpt(self.tprob)
        all_to_all_mfpt_ref = io.loadh(os.path.join(self.tpt_ref_dir, "all_to_all_mfpt.h5"), 'Data')
        npt.assert_array_almost_equal(all_to_all_mfpt, all_to_all_mfpt_ref)
        
        
    def test_TP_time(self):
        tp_time = tpt.calculate_avg_TP_time(self.sources, self.sinks, self.tprob, self.lag_time)
        tp_time_ref = io.loadh(os.path.join(self.tpt_ref_dir, "tp_time.h5"), 'Data')
        npt.assert_array_almost_equal(tp_time, tp_time_ref)
        
        
    def test_fraction_visits(self):
        
        num_to_test = 11**2 # this can be changed to shorten the test a little
        
        for i in range(num_to_test):
            waypoint = int(self.hc[i,0])
            source   = int(self.hc[i,1])
            sink     = int(self.hc[i,2])
            hc = tpt.calculate_fraction_visits(self.hub_T, waypoint, source, sink)
            assert np.abs(hc - self.hc[i,3]) < 0.0001
        

    def test_hub_scores(self):
        
        all_hub_scores = tpt.calculate_all_hub_scores(self.hub_T)
        npt.assert_array_almost_equal( all_hub_scores, self.Hc )
        
        for waypoint in range(self.hub_T.shape[0]):
            hub_score = tpt.calculate_hub_score(self.hub_T, waypoint)
            assert np.abs(hub_score - all_hub_scores[waypoint]) < 0.0001
        
        
