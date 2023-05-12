import NormalizingFlows as nf
import torch
from torch.distributions import Independent, Uniform
from Simulation import *
import numpy


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

D = 5

a = torch.tensor([1,2,3,1,2]).to(device)

exact_l = 1339/1440


def target_weight(x):
    x = x*a
    x1 = x[...,0]
    x2 = x[...,1]
    x3 = x[...,2]
    x4 = x[...,3]
    x5 = x[...,4]
    h = torch.stack((x1+x4,x1+x3+x5,x2+x3+x4,x2+x5), dim = -1).min(-1).values
    return h

d = Independent(Uniform(torch.zeros(D).to(device), torch.ones(D).to(device)),1)

perm = torch.cat([torch.arange(D//2,D),torch.arange(D//2)])
flows = []
for i in range(5):
    flows.append(nf.CouplingFlow(D, normalise=True, N=1))
    flows.append(nf.PermutationFlow(perm))


net = nf.NormFlows(flows, d).to(device)

trainer = Trainer(net, device = device, bs = 10000)
results = Results(net, plot_ss = 10000, eval_ss = 10000)

train_net(
    net, trainer, results, 300000, target_weight,
    weight_log = False,
    criterion = "KL1",
    penalty = None,
    lr = 0.0001,
    wd = 0.0001,
    print_interval = 10,
    last_only = False,
    save_dir = "experiments/bridge_network")



z = net.sample(10000)
logpy, y = net.logpy(z = z, return_zy = True)
path = target_weight(y)
print(target_weight(z).std(True).item())
print((path/logpy.exp()).std(True).item())
print(target_weight(z).mean().item())
print((path/logpy.exp()).mean().item())


#Estimate KL divergence
target_logp = (path/exact_l).log()
log_rat = target_logp - logpy
KL_samples = log_rat.exp()*log_rat
print(KL_samples.mean().item(),KL_samples.std(unbiased = True).item()/numpy.sqrt(len(KL_samples)))

