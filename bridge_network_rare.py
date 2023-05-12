import NormalizingFlows as nf
import torch
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

def penalty(x):
    x = x*a
    x1 = x[...,0]
    x2 = x[...,1]
    x3 = x[...,2]
    x4 = x[...,3]
    x5 = x[...,4]
    mid = torch.stack((x1+x3+x5,x2+x3+x4), dim = -1).min(-1).values
    out = torch.stack((x1+x4,x2+x5), dim = -1).min(-1).values
    return mid - out
    
# load the untrained model
net = torch.load("experiments/bridge_network/checkpoints/net-0.pt").to(device)

trainer = Trainer(net, device = device, bs = 10000, alpha = 10)
results = Results(net, plot_ss = 10000, eval_ss = 10000)

results.plot_z = torch.load("experiments/bridge_network/plot_z.pt").to(device)


train_net(
    net, trainer, results, 500000, target_weight,
    weight_log = False,
    criterion = "KL1",
    penalty = penalty,
    lr = 0.0001,
    wd = 0.0001,
    print_interval = 10,
    last_only = False,
    save_dir = "experiments/bridge_network_rare")


z = net.sample(10000)
logpy, y = net.logpy(z = z, return_zy = True)
path = target_weight(y)



summand = torch.where(penalty(y)<=0, path/logpy.exp(), trainer.zero) # estimate l
summand2= torch.where(penalty(y)<=0, 1/logpy.exp(), trainer.zero) # estimate c

c = summand2.mean().item()
l = summand.mean().item()
print(summand2.std(True).item())
print(summand.std(True).item())
print(c)
print(l/c)


#Estimate KL divergence
target_logp = torch.where(penalty(y)<=0,(path/(l)).log(), trainer.zero)
log_rat = target_logp - logpy
KL_samples = log_rat.exp()*log_rat
print(KL_samples.mean().item(),KL_samples.std(unbiased = True).item()/numpy.sqrt(len(KL_samples)))

