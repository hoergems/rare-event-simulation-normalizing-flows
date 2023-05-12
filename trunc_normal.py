import NormalizingFlows as nf
import torch
from torch.distributions import Independent, Normal
from Simulation import * # imports Trainer, Results and train_net

# Set device to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# base distribution
d = Independent(Normal(torch.zeros(1).to(device),torch.ones(1).to(device)),1)

# model
net = nf.NonlinearSquaredFlowSeq(N = 3, noise_distribution = d).to(device)

# trainer and results objects
trainer = Trainer(net, device = device,
                  alpha = 100, # penalty parameter
                  bs = 1000) # training batch size
results = Results(net,
                  plot_ss = 1000, # plotting sample size
                  eval_ss = 1000) # evaluating sample size

gamma = 3 # rarity level

# theoretical rare event probability
c = 1-0.5*(1+torch.erf(torch.tensor(3)/numpy.sqrt(2))).item()


# the target_weight function should return a tensor of shape [batch size]
def target_weight(y):
    """ The target function.
    The input should be a tensor of shape [batch size, num dims].
    The output should either be the target or the log of the target evaluated
    at each batch point. In this case the log of the unconditional target
    density is returned. The output should have shape [batch size].
    """
    return d.log_prob(y)

# gamma - S(x)
def penalty(y):
    """This function should return gamma - S(x).
    Therefore, if the returned value is positive, then the point is not in the
    rare-event region.
    The input should be a tensor of shape [batch size, num dims].
    The output should be a tensor of shape [batch size].
    """
    return gamma-y.squeeze()

# train the model for 30000 iterations
train_net(
    net, trainer, results, 30000, target_weight,
    weight_log = True, # True since target_weight returns the log density 
    criterion = "KL1", # KL divergence objective
    penalty = penalty, # penalty function
    lr = 0.001, # learning rate
    wd = 0.0001, # weight decay
    print_interval = 100, # print loss every 100 iterations
    last_only = False, # save every checkpoint (not only the last one)
    save_dir = "experiments/trunc_normal")



# estimators
z = net.sample(1000)
logpy, y = net.logpy(z = z, return_zy = True)

summand = torch.where(penalty(y)<=0, (target_weight(y)-logpy).exp(), trainer.zero)
print(summand.mean().item())
print(summand.std(True).item())


#Estimate KL divergence between learnt and target distributions
z = net.sample(10000)
logpy, y = net.logpy(z = z, return_zy = True)
logweight = target_weight(y)
log_rat = logweight - logpy
ind = penalty(y)<0
KL_samples = torch.where(ind,log_rat.exp()*(log_rat-numpy.log(c))/c,torch.tensor(0.0).to(device))
print(KL_samples.mean().item(),KL_samples.std(unbiased = True).item()/numpy.sqrt(len(KL_samples)))



