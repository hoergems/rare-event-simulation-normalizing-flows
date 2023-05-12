# Rare-Event Simulation and Normalizing Flows

This repository contains the source code for our project on learning normalizing flows based generative models for rare event simulations.
The easiest example to follow is trunc_normal.py.

## Files
| File | Description |
| ------ | ------ |
| NormalizingFlows.py | Core library for building normalizing flow models and is well documented |
| Simulation.py | Supporting library for training models |
| trunc_normal.py | 1D truncated normal distribution example. Very simple and commented. Read this first. |
| sum_exp_gamma10.py | 2D truncated exponenetial distribution example. |
| bridge_network.py | 5D bridge network example. |
| bridge_network_rare.py | Bridge network rare-event example. |
| asian_option.py | 88D Asian option example. |
| asian_option_rare.py | Asian option rare-event example. |
| double_slit.py | Double slit rare-event example. |


## Required Libraries

- PyTorch
- NumPy
- Matplotlib

## General Training Approach
- Import libraries using:
    - import torch
    - import NormalizingFlows as nf
    - from Simulation import *
- Construct the base distribution object.
    - torch.distributions is very useful.
    - for example, to make a 10D multivariate normal distirbution you can use: d = Independent(Normal(torch.zeros(10).to(device),torch.ones(10).to(device)),1)
- Construct the model as an instance of NormFlows from NormalizingFlows.py. For example:
    - net = nf.NonlinearSquaredFlowSeq(N = 3, noise_distribution = d).to(device)
- Construct "Trainer" and "Results" objects from Simulation.py.
    - includes parameters like the learning rate, and the penalty weight (alpha).
- Define target_weight and penalty functions.
    - These are used to compute h(x) (or ln[h(x)]), and gamma - S(x)
- Train the model using the train_net function from Simulation.py.

## Contributors
- [Dr. Lachlan Gibson](mailto:l.gibson1@uq.edu.au), School of Mathematics & Physics, The University of Queensland, Australia
- [Dr. Marcus Hoerger](mailto:m.hoerger@uq.edu.au), School of Mathematics & Physics, The University of Queensland, Australia
- [Prof. Dirk Kroese](mailto:kroese@maths.uq.edu.au), School of Mathematics & Physics, The University of Queensland, Australia 
