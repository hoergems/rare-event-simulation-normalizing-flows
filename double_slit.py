import torch
import argparse
import os
import NormalizingFlows as nf
import numpy as np
import pylab as pl
from torch.distributions import Independent, Normal, Categorical, MixtureSameFamily
from Simulation import * # imports Trainer, Results and train_net
from matplotlib import collections as mc

save_dir = "experiments/double_slit"

# Use this if training slows down over time
torch.set_flush_denormal(True)

# Set device to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#Parameters for the double slit environment
initial_position = [0, 0]
slit_x = 5.0
slit_y = 1.5
slit_w = 1.0
screen = 10.0

class GMM():
	def __init__(self, mu, sigma):		
		self._dimensions = mu.shape[1]
		mix = Categorical(torch.ones(mu.shape[0],))
		comp = Independent(Normal(mu, sigma), 1)
		self._gmm = MixtureSameFamily(mix, comp)	

	def sample(self, sample_shape):		
		return self._gmm.sample(sample_shape).reshape((sample_shape[0], self._dimensions))

	def log_prob(self, z):		
		return self._gmm.log_prob(z)

class DTrainer():
    def __init__(self, net, device = None, alpha = 10, beta = 10, bs = 10000, save_iteration=100):
        self.iteration = 0
        self.alpha = alpha
        self.beta = beta        
        self.bs = bs
        self.save_checkpoints = True
        self.save_iteration = save_iteration
        self.image_num = 0
        
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        
        self.device = device
        self.zero = torch.tensor(0.0).to(device)
    
    def eval_condition(self):
        if not self.save_checkpoints:
            return False        
        c = self.iteration % self.save_iteration == 0        
        return c

class DSResults:
    def __init__(self, net, D, plot_ss = 10000, eval_ss = 10000, rescale = False):
    	self._D = D
    	self._results = []

    def save(self, folder = ""):
    	path = os.path.join(folder, "success_rate_D_" + str(self._D) + ".csv")
    	numpy.savetxt(path,np.array(self._results),fmt='%i, %1.4f', header='iteration,success_rate',delimiter=',')       

    def load(self, folder = ""):
        pass

    def eval_net(self, net, trainer, target_weight,
                 weight_log = False, penalty = None, criterion = "ISstd"):    	
    	net.eval()
    	with torch.no_grad():
        	# Compute success rate
        	logpy, y = net.logpy(z = net.sample(10000), return_zy = True)        	
        	positions = torch.tensor(initial_position).to(device) + torch.cumsum(y.view((y.shape[0], (int)(y.shape[1]/2), 2)), 1)
        	x_positions = positions[:, :, 0:1]
        	y_positions = positions[:,:,1:]

        	ind = x_positions>slit_x# logical indices where x coord is right of the slits
	        ind = ind.squeeze().reshape((x_positions.shape[0], x_positions.shape[1]))
	        ind2 = torch.where(ind, 1, 0).reshape((x_positions.shape[0], x_positions.shape[1]))
	                  
	        first_nonzero = ind2.argmax(1)
	        ind[:] = False
	        ind[tuple(torch.stack([torch.arange(ind2.shape[0]), first_nonzero.t().squeeze().reshape((x_positions.shape[0]))]))] = True
	        ind[tuple(torch.stack([torch.arange(ind2.shape[0], dtype=torch.int64), torch.zeros(ind.shape[0], dtype=torch.int64)]))] = False
	        ind = ind.reshape((ind.shape[0], ind.shape[1], 1)) 

	        y_positions_slit = torch.where(ind, y_positions, -float('inf'))
	        y_positions_slit = (y_positions_slit.max(dim=1).values.abs() - slit_y).abs()
	        success_rate_slit = torch.where(y_positions_slit <= slit_w/2, 1, 0).sum(dim=0)/x_positions.shape[0] 

	        ind = x_positions>screen# logical indices where x coord is right of the screen
	        ind = ind.squeeze().reshape((x_positions.shape[0], x_positions.shape[1]))
	        ind2 = torch.where(ind, 1, 0).reshape((x_positions.shape[0], x_positions.shape[1]))
	                  
	        first_nonzero = ind2.argmax(1)
	        ind[:] = False
	        ind[tuple(torch.stack([torch.arange(ind2.shape[0]), first_nonzero.t().squeeze().reshape((x_positions.shape[0]))]))] = True
	        ind[tuple(torch.stack([torch.arange(ind2.shape[0], dtype=torch.int64), torch.zeros(ind.shape[0], dtype=torch.int64)]))] = False
	        ind = ind.reshape((ind.shape[0], ind.shape[1], 1))

	        y_positions_screen = torch.where(ind, y_positions, -float('inf'))	        
	        y_positions_screen = y_positions_screen.max(dim=1).values        
	        success_rate_screen = torch.where(y_positions_screen > np.NINF, 1, 0).sum(dim=0)/x_positions.shape[0]

	        self._results.append([trainer.iteration, torch.stack((success_rate_slit, success_rate_screen)).min(dim=0).values.numpy()[0]])
    def eval_z(self, z, net, trainer, target_weight,
               weight_log = False, penalty = None, criterion = "ISstd"):
        pass

class TanhAct(torch.nn.Module):
    def __init__(self, alpha):
        super(TanhAct, self).__init__()
        self.alpha = alpha        

    def forward(self, x):        
        return self.alpha * torch.tanh(x)

def make_net(D, sampling_distribution, n_layers=20, tanh_const=7.0):
    perm = torch.cat([torch.arange(D//2,D),torch.arange(D//2)])
    flows = []
    for i in range(n_layers):
        flows.append(nf.CouplingFlow(D, N = 1, parameter_max_bound = 0.00001, HLsizes=[], outact=TanhAct(tanh_const)))
        flows.append(nf.PermutationFlow(perm))
    return nf.NormFlows(flows, sampling_distribution).to(device)

def make_base_distribution(D, variance_scale):
    mu1 = torch.ones(D).to(device)
    mu2 = torch.ones(D).to(device)
    for i in range(D):
        if i % 2 != 0:
            mu2[i] *= -1    

    return GMM(torch.stack((mu1, mu2)), torch.stack((variance_scale*torch.ones(D).to(device), variance_scale*torch.ones(D).to(device))))

class DoubleSlitExperiment:
    def __init__(self, args):        
        self._num_plotted_paths = args.num_plotted_paths
        self._T = 1.0
        self._D = args.num_dims
        self._variance_scale = np.sqrt((2.0*self._T / (self._D)))
        # Base distribution
        mu1 = torch.ones(self._D).to(device)
        mu2 = torch.ones(self._D).to(device)
        for i in range(self._D):
        	if i % 2 != 0:
        		mu2[i] *= -1
        self._base_distr = GMM(torch.stack((mu1, mu2)), 
        	torch.stack((self._variance_scale*torch.ones(self._D).to(device), self._variance_scale*torch.ones(self._D).to(device))))
        # Target distribution
        self._target_distr = Independent(Normal(torch.zeros(self._D), self._variance_scale*torch.ones(self._D)), 1)            
        # construct coupling flows model        
        self._net = make_net(self._D, self._base_distr)
        # trainer and results objects
        self._trainer = DTrainer(self._net, device = device,                                
                                alpha = 100.0, # penalty parameter
                                beta=0.0,
                                bs = args.batch_size, # training batch size
                                save_iteration=args.save_iteration) 
        self._results = DSResults(self._net, self._D)
        self._trainer.save_checkpoints = True

    def target_weight(self, y):            
        return self._target_distr.log_prob(y)    

    def penalty(self, y):
        positions = torch.tensor(initial_position).to(device) + torch.cumsum(y.view((y.shape[0], (int)(y.shape[1]/2), 2)), 1)  
        x_positions = positions[:, :, 0:1]    
        y_positions = positions[:,1:,1:]    

        # Penalty for not hitting the screen (Filter out points where x > x_screen)
        pen_screen = screen - x_positions.max(1).values
        pen_screen = torch.where(pen_screen > 0, pen_screen, 0.0)

        ind = x_positions>slit_x# logical indices where x coord is right of slits
        ind = torch.logical_xor(ind[:,1:],ind[:,:-1])
        
        y_positions1 = y_positions-slit_y
        y_positions1 = torch.where(ind, y_positions1, 0.0).abs()
        ind_y = y_positions1 > 0.5*slit_w
        pen_slit1 = torch.where(ind_y, (y_positions1-0.5*slit_w), 0.0).max(1).values# - 0.5*slit_w

        y_positions2 = y_positions+slit_y
        y_positions2 = torch.where(ind, y_positions2, 0.0).abs()
        ind_y = y_positions2 > 0.5*slit_w    
        pen_slit2 = torch.where(ind_y, (y_positions2-0.5*slit_w), 0.0).max(1).values        
        pen_slit = torch.stack((pen_slit1, pen_slit2), dim=1).min(dim=1).values        
        return pen_slit + pen_screen

    def plot_fig(self, net, trainer, results, image_dir):
        with torch.no_grad():
            color_red = [(1, 0, 0, 1)]
            color_goal = [(0, 1, 0, 1)]
            color_barriers = [(0, 0, 0, 1)]
            color_blue = [(0, 0, 1, 1)]

            fig, axes = pl.subplots(1, 2)

            # Line collections for the barriers
            plot_y_lim = 10.0
            barrier1 = np.array([slit_x, slit_y+0.5*slit_w, slit_x, plot_y_lim]).reshape(2, 2)
            barrier2 = np.array([slit_x, -slit_y-0.5*slit_w, slit_x, -plot_y_lim]).reshape(2, 2)
            barrier3 = np.array([slit_x, -(slit_y-0.5*slit_w), slit_x, slit_y-0.5*slit_w]).reshape(2, 2)            
            barriers_array = np.stack((barrier1, barrier2, barrier3))
            line_collection_barriers = mc.LineCollection(barriers_array, colors=color_barriers, linewidths=4)
            axes[0].add_collection(line_collection_barriers)

            # Line collection for the screen
            screen_array = np.array([screen, plot_y_lim, screen, -plot_y_lim]).reshape(1, 2, 2)
            line_collection_screen = mc.LineCollection(screen_array, colors=color_goal, linewidths=4)
            axes[0].add_collection(line_collection_screen)

            logpy, y = net.logpy(z = net.sample(self._num_plotted_paths), return_zy = True)
            init_pos = torch.tensor(initial_position).to(device)
            positions = init_pos + torch.cumsum(y.view((y.shape[0], (int)(y.shape[1]/2), 2)), 1)

            # Line collection for the paths
            for j in range(positions.shape[0]):
                pos = torch.cat((init_pos.view(1, 2), positions[j]))                                            
                line_collection_array = pos.reshape(1, pos.shape[0], 2)            
                line_collection = mc.LineCollection(line_collection_array.detach(), colors=color_red, linewidths=1)
                axes[0].add_collection(line_collection)                


            # Make histogram        
            x_positions = positions[:, :, 0:1]
            y_positions = positions[:,:,1:]
            ind = x_positions>screen# logical indices where x coord is right of screen
            ind = ind.squeeze().reshape((x_positions.shape[0], x_positions.shape[1]))
            ind2 = torch.where(ind, 1, 0).reshape((x_positions.shape[0], x_positions.shape[1]))

            # Get index of the first point that hits the screen
            first_nonzero = ind2.argmax(1)
            ind[:] = False
            ind[tuple(torch.stack([torch.arange(ind2.shape[0]), first_nonzero.t().squeeze().reshape((x_positions.shape[0]))]))] = True
            ind[tuple(torch.stack([torch.arange(ind2.shape[0], dtype=torch.int64), torch.zeros(ind.shape[0], dtype=torch.int64)]))] = False
            ind = ind.reshape((ind.shape[0], ind.shape[1], 1))
            
            y_positions_screen = torch.where(ind, y_positions, -float('inf'))
            
            y_positions_np = y_positions_screen.max(dim=1).values.detach().squeeze().numpy()        
            axes[1].hist(y_positions_np, 
                density = True,
                orientation = "horizontal",
                bins = 50, 
                range = (-10,10),
                alpha=0.5,
                color=[color_red[0]])

            axes[0].set_xlim(-10, 10)
            axes[0].set_ylim(-10, 10)
            axes[1].set_xlim(0, 0.4)
            axes[1].set_ylim(-10, 10)
            axes[1].axes.yaxis.set_visible(False)        
            pl.margins(0)
            pl.savefig(os.path.join(image_dir, "double_slit_" + str(self._D) + "_" + str(trainer.iteration) + ".png"), dpi=199, bbox_inches='tight')
            pl.close()        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load_model', action='store_true')
    parser.add_argument('-D', '--num_dims', type=int, default=100, help="Number of dimensions. num_dims/2 corresponds to the number of time steps. Must be an even number")
    parser.add_argument('-i', '--iterations', type=int, default=10000, help="Number of training iterations")
    parser.add_argument('-s', '--save_iteration', type=int, default=1000, help="Save the model every save_iteration iterations")
    parser.add_argument('-b', '--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('-p', '--num_plotted_paths', type=int, default=100, help="Number of plotted paths")
    args = parser.parse_args()

    assert args.num_dims > 2 and args.num_dims % 2 == 0, "num_dims must be greater than 2 and an even number"

    double_slit_experiment = DoubleSlitExperiment(args)       
    train_net(double_slit_experiment._net, double_slit_experiment._trainer, double_slit_experiment._results, args.iterations, double_slit_experiment.target_weight,
        weight_log = True, # True since target_weight returns the log density
        plot_fig=double_slit_experiment.plot_fig,
        criterion = "KL1", # KL divergence objective
        penalty = double_slit_experiment.penalty, # penalty function
        lr = 1e-6, # learning rate
        wd = 1e-8, # weight decay
        grad_norm=0.9, # Clip gradient norm
        print_interval = 1, # print loss every 100 iterations
        last_only = False, # save every checkpoint (not only the last one)
        save_dir = save_dir)    