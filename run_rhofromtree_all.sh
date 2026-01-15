#!/bin/bash

# Check that ANG1 is provided
if [ $# -lt 2 ]; then
  echo "Usage: $0 <ANG1> <ANG2>"
  exit 1
fi

ANG1=$1
ANG2=$2

HISTCFG="/nfs/cms/arqolmo/ExamplesFCCFullSim/config/histograms/rho_analysis_config.yml"
CFG="/nfs/cms/arqolmo/ExamplesFCCFullSim/config/default/taurecolong_optimal.yaml"

nohup python RhoAnalysis/rhoHistFromTree.py -d 2 \
  --hist-config "$HISTCFG" \
  --tree-file "/nfs/cms/arqolmo/ExamplesFCCFullSim/Results/RhoAnalysis/Zqq_sampletau_trained0.4_tph0.35_tpi0_n3_g0.0/tau_traineddecay2_0.4_tph0.35_tpi0_n3_g0.0.root" \
  -v -c "$CFG" --ang "$ANG1" "$ANG2" &

nohup python RhoAnalysis/rhoHistFromTree.py -d 0 \
  --hist-config "$HISTCFG" \
  --tree-file "/nfs/cms/arqolmo/ExamplesFCCFullSim/Results/RhoAnalysis/Zqq_sampletau_trained0.4_tph0.35_tpi0_n3_g0.0/tau_traineddecay0_0.4_tph0.35_tpi0_n3_g0.0.root" \
  -v -c "$CFG" --ang "$ANG1" "$ANG2" &

nohup python RhoAnalysis/rhoHistFromTree.py -d 2 \
  --hist-config "$HISTCFG" \
  --tree-file "Results/RhoAnalysis/Zee_sample_tau_trained0.4_tph0.35_tpi0_n3_g0.0/tau_traineddecay2_0.4_tph0.35_tpi0_n3_g0.0.root" \
  -v -c "$CFG" --ang "$ANG1" "$ANG2" &

nohup python RhoAnalysis/rhoHistFromTree.py -d 0 \
  --hist-config "$HISTCFG" \
  --tree-file "Results/RhoAnalysis/Zee_sample_tau_trained0.4_tph0.35_tpi0_n3_g0.0/tau_traineddecay0_0.4_tph0.35_tpi0_n3_g0.0.root" \
  -v -c "$CFG" --ang "$ANG1" "$ANG2" &

nohup python RhoAnalysis/rhoHistFromTree.py -d 2 \
  --hist-config "$HISTCFG" \
  --tree-file "Results/RhoAnalysis/bhabha_sample_tau_trained0.4_tph0.35_tpi0_n3_g0.0/tau_traineddecay2_0.4_tph0.35_tpi0_n3_g0.0.root" \
  -v -c "$CFG" --ang "$ANG1" "$ANG2" &

nohup python RhoAnalysis/rhoHistFromTree.py -d 0 \
  --hist-config "$HISTCFG" \
  --tree-file "Results/RhoAnalysis/bhabha_sample_tau_trained0.4_tph0.35_tpi0_n3_g0.0/tau_traineddecay0_0.4_tph0.35_tpi0_n3_g0.0.root" \
  -v -c "$CFG" --ang "$ANG1" "$ANG2" &

nohup python RhoAnalysis/rhoHistFromTree.py -d 2 \
  --hist-config "$HISTCFG" \
  --tree-file "GATr_Results/RhoAnalysis/PFO_tau_trained0.4_tph0.35_tpi0_n3_g0.0/tau_traineddecay2_0.4_tph0.35_tpi0_n3_g0.0.root" \
  -v -c "$CFG" --ang "$ANG1" "$ANG2" &

nohup python RhoAnalysis/rhoHistFromTree.py -d 0 \
  --hist-config "$HISTCFG" \
  --tree-file "GATr_Results/RhoAnalysis/PFO_tau_trained0.4_tph0.35_tpi0_n3_g0.0/tau_traineddecay0_0.4_tph0.35_tpi0_n3_g0.0.root" \
  -v -c "$CFG" --ang "$ANG1" "$ANG2" &

nohup python RhoAnalysis/rhoHistFromTree.py -d 2 \
  --hist-config "$HISTCFG" \
  --tree-file "Results/RhoAnalysis/tau_trained0.4_tph0.35_tpi0_n3_g0.0/tau_traineddecay2_0.4_tph0.35_tpi0_n3_g0.0.root" \
  -v -c "$CFG" --ang "$ANG1" "$ANG2" &

nohup python RhoAnalysis/rhoHistFromTree.py -d 0 \
  --hist-config "$HISTCFG" \
  --tree-file "Results/RhoAnalysis/tau_trained0.4_tph0.35_tpi0_n3_g0.0/tau_traineddecay0_0.4_tph0.35_tpi0_n3_g0.0.root" \
  -v -c "$CFG" --ang "$ANG1" "$ANG2" &

wait
echo "All jobs launched with ANG1=${ANG1} ANG2=${ANG2}"