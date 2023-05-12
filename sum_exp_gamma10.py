import NormalizingFlows as nf
import torch
from torch.distributions import Independent, Normal
from Simulation import *
import numpy

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# two dimensions
D = 2

# base distribution is multivariate normal
d = Independent(Normal(torch.zeros(D).to(device),torch.ones(D).to(device)),1)

# construct coupling flows model
perm = [1,0] # permutation that swaps dimensions
flows = []
for i in range(6):
    flows.append(nf.CouplingFlow(D, N=2))
    flows.append(nf.PermutationFlow(perm))
flows.append(nf.ElementwiseExpFlow())
net = nf.NormFlows(flows, d).to(device)


trainer = Trainer(net, device = device, alpha = 100)
results = Results(net)

gamma = 10
thc = (1+gamma)*numpy.exp(-gamma)

def target_weight(y):
    return -y.sum(-1)

def penalty(y):
    return gamma-y.sum(-1)

train_net(
    net, trainer, results, 100000, target_weight,
    weight_log = True,
    criterion = "KL1",
    penalty = penalty,
    lr = 0.0001,
    wd = 0.0001,
    print_interval = 10,
    save_dir = "experiments/sum_exp_gamma10")



z = net.sample(1000)
logpy, y = net.logpy(z = z, return_zy = True)

#Estimate probability of rare event
logweight = target_weight(y)
log_rat = logweight - logpy
ind = penalty(y)<0
IS_samples = torch.where(ind,log_rat.exp(),torch.tensor(0.0).to(device))
IS_m = IS_samples.mean()
IS_s = IS_samples.std(unbiased = True)
RE = (IS_s/IS_m/numpy.sqrt(y.shape[0])).item()
m = IS_m.item()
s = IS_s.item()

print("m",m)
print("s",s)
print("RE",RE)
print("found valid trajectories",ind.sum().item())


#Estimate KL divergence
z = net.sample(10000)
logpy, y = net.logpy(z = z, return_zy = True)
logweight = target_weight(y)
log_rat = logweight - logpy
ind = penalty(y)<0
KL_samples = torch.where(ind,log_rat.exp()*(log_rat-numpy.log(thc))/thc,torch.tensor(0.0).to(device))
print(KL_samples.mean().item(),KL_samples.std(unbiased = True).item()/numpy.sqrt(len(KL_samples)))



