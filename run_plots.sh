#!/bin/bash
python TauAnalysis/TausCompletePlot.py -i GATr_Results/TauReco/tau_trained0.4_tph0.35_tpi0.6_n0.0_g0.0/ -p config/plots/taulong_plotconfig_results.yaml
python TauAnalysis/TausCompletePlot.py -i  GATr_Results/TauReco/PFO_tau_trained0.4_tph0.35_tpi0_n3_g0.0/ -p config/plots/taulong_plotconfig_results.yaml