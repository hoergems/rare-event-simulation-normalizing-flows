import NormalizingFlows as nf
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal
from Simulation import *
import numpy


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


r = 0.07 # annual interest
sig = 0.2 # volatility
K = 35 # strike price
S_0 = 40 # initial stock price
T = 4/12 #maturity in 4 months, which is 4/12 of the year
n = 88 # there are approx. 88 trading days in 4 months
dt = T/n # time step
sqrtdt = numpy.sqrt(dt)
disc = numpy.exp(-r*T)

gamma = 14

t = torch.linspace(0,T,n+1).to(device)
drift = (r-sig**2/2)*t

D = n

d = Independent(Normal(torch.zeros(D).to(device),sqrtdt*torch.ones(D).to(device)),1)

perm = torch.cat([torch.arange(D//2,D),torch.arange(D//2)])
flows = []
for i in range(6):
    flows.append(nf.CouplingFlow(D, N = 1, parameter_max_bound = 0.0001))
    flows.append(nf.PermutationFlow(perm))

net = nf.NormFlows(flows, d).to(device)



trainer = Trainer(net, device = device, bs = 1000)
results = Results(net, plot_ss = 1000, eval_ss = 1000)


def smooth_relu(x, thresh = 1, log = False):
    # A positive smooth piecewise function to approximate relu
    # f(x) = x when x >= t, and f(x) = t*exp(x/t-1) when x < t,
    # where t is thresh.
    ind = x < thresh
    if log:
        return torch.where(ind,x/thresh-1+numpy.log(thresh),x.log())
    else:
        return torch.where(ind,thresh*torch.exp(x/thresh-1),x)


def trajectories(x):
    # x should be a tensor of shape [batch dims, n]
    W = (sig*x).cumsum(-1)
    S = S_0*torch.exp(drift[1:]+W)
    return S

def target_weight(x):
    return d.log_prob(x)

def penalty(x):
    S = trajectories(x)
    Sbar = (S_0 + S.sum(-1))/(n+1)
    return gamma - disc*(Sbar-K)

def payoff(x):
    S = trajectories(x)
    Sbar = (S_0 + S.sum(-1))/(n+1)
    return disc*(Sbar-K).relu()


for i in range(10):
    train_net(
        net, trainer, results, 10000, target_weight,
        weight_log = True, # True if output of target_weight is log density
        criterion = "KL1",
        penalty = penalty,
        lr = 0.0001,
        wd = 0.0001,
        print_interval = 10,
        last_only = True, # only save last checkpoint
        save_dir = "experiments/asian_option_rare")


# Estimators
z = net.sample(10000)
logpy, y = net.logpy(z = z, return_zy = True)
target = target_weight(y)
ind = penalty(y) > 0
estimator = torch.exp(d.log_prob(y)-logpy)
estimator[ind] = 0
l = estimator.mean().item()
RE = estimator.std(unbiased = True).item()/numpy.sqrt(z.shape[0])/l

# crude monte carlo
crude_estimator = 1.0*(penalty(z) <= 0)
crude_l = crude_estimator.mean().item()
crude_RE = crude_estimator.std(unbiased = True).item()/numpy.sqrt(z.shape[0])/crude_l

# KL divergence
target_logp = torch.where(penalty(y)<=0,target-numpy.log(l), trainer.zero)
log_rat = target_logp - logpy
KL_samples = log_rat.exp()*log_rat
print(KL_samples.mean().item(),KL_samples.std(unbiased = True).item()/numpy.sqrt(len(KL_samples)))


