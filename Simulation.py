import torch
import numpy
import os


class Trainer():
    def __init__(self, net, device = None, alpha = 10, bs = 10000):
        self.iteration = 0
        self.alpha = alpha
        self.bs = bs
        self.save_checkpoints = True
        self.log_interval = numpy.log10(100000)/(30*20)
        self.log_track = -1-self.log_interval
        self.image_num = 0
        
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        
        self.device = device
        self.zero = torch.tensor(0.0).to(device)
        self.negone = torch.tensor(-1.0).to(device)
    
    def eval_condition(self):
        if not self.save_checkpoints:
            return False
        n_track = numpy.log10(1+self.iteration)
        c = (n_track-self.log_track) >= self.log_interval
        if c:
            self.log_track = n_track
        return c


class Results():
    def __init__(self, net, plot_ss = 10000, eval_ss = 10000, rescale = False):
        self.plot_z = net.sample(plot_ss)
        self.plot_ss = plot_ss
        self.eval_ss = eval_ss
        self.t = []
        self.plot_z_results = []
        self.rand_z_results = []
        self.labels=[
            "IS_s","IS_m","RE","pen_loss","loss","KL1","KL2","logstd",
            "log_scale"]
        self.header = "IS_s,IS_m,RE,pen_loss,loss,KL1,KL2,logstd,log_scale"
        self.plot_z_saved = False
        self.rescale = rescale
    
    def save(self, folder = ""):
        if not self.plot_z_saved:
            torch.save(self.plot_z,os.path.join(folder, "plot_z.pt"))
            self.plot_z_saved = True
        plot_z_results = numpy.asarray(self.plot_z_results)
        rand_z_results = numpy.asarray(self.rand_z_results)
        t = numpy.asarray(self.t)
        numpy.savetxt(
            os.path.join(
                folder, "plot_z_results.csv"),
            plot_z_results, delimiter=",", header = self.header,
            fmt='%f',comments='')
        
        numpy.savetxt(
            os.path.join(
                folder, "rand_z_results.csv"),
            rand_z_results, delimiter=",", header = self.header,
            fmt='%f',comments='')
        
        numpy.savetxt(os.path.join(folder, "t.csv"),t,fmt='%d')
    
    def load(self, folder = ""):
        self.plot_z = torch.load(os.path.join(folder, "plot_z.pt"))
        self.plot_z_results = numpy.genfromtxt(
            os.path.join(
                folder, "plot_z_results.csv"),
            delimiter = ",",
            skip_header = 1,
            ).tolist()
        self.rand_z_results = numpy.genfromtxt(
            os.path.join(
                folder, "rand_z_results.csv"),
            delimiter = ",",
            skip_header = 1,
            ).tolist()
        self.t = numpy.genfromtxt(os.path.join(folder, "t.csv")).tolist()
    
    def eval_net(self, net, trainer, target_weight,
                 weight_log = False, penalty = None, criterion = "ISstd"):
        net.eval()
        with torch.no_grad():
            z = net.sample(self.eval_ss)
            r1 = self.eval_z(z, net, trainer, target_weight,
                             weight_log = weight_log, penalty = penalty,
                             criterion = criterion)
            r2 = self.eval_z(
                self.plot_z, net, trainer, target_weight,
                weight_log = weight_log, penalty = penalty,
                criterion = criterion)
        self.t.append(trainer.iteration)
        self.plot_z_results.append(list(r2))
        self.rand_z_results.append(list(r1))
    
    def eval_z(self, z, net, trainer, target_weight,
               weight_log = False, penalty = None, criterion = "ISstd"):
        logpy, y, dgdz = net.logpy(z = z, return_zy = True, return_jac = True)
        if weight_log:
            logweight = target_weight(y)
            weight = logweight.exp()
        else:
            weight = target_weight(y)
            logweight = weight.log()
        log_diff = logpy - logweight
        KL1 = log_diff.mean().item()
        logstd = log_diff.std(unbiased = True).item()
        pen_loss = 0
        if penalty is not None:
            pen = penalty(y)
            ind = pen>=0
            pen_loss = torch.where(
                ind,pen*trainer.alpha, trainer.zero).mean().item()
            weight = torch.where(~ind, weight, trainer.zero)
            IS_m, IS_s, log_scale, IS_samples = IS_scaled(
                logweight, logpy, trainer, ind=~ind, rescale = self.rescale)
        else:
            IS_m, IS_s, log_scale, IS_samples = IS_scaled(
                logweight, logpy, trainer, rescale = self.rescale)
        
        RE = (100*IS_s/IS_m/numpy.sqrt(z.shape[0])).item()
        IS_m = IS_m.item()
        IS_s = IS_s.item()
        
        if criterion == "ISstd":
            loss = IS_s + pen_loss
        elif criterion == "KL1":
            loss = KL1 + pen_loss
        elif criterion == "logstd":
            loss = logstd + pen_loss
        elif criterion == "KL_simp":
            loss = -(logweight+dgdz).mean().item() + pen_loss
        
        KL2 = (logweight - logpy)*IS_samples
        KL2 = torch.where(KL2.isnan(),trainer.zero, KL2).mean().item()
        
        return IS_s, IS_m, RE, pen_loss, loss, KL1, KL2, logstd, log_scale
        

def train_net(
        net, trainer, results, nepochs, target_weight,
        weight_log = False,
        plot_fig = None,
        penalty = None,
        lr = 0.001,
        wd = 0.0001,
        grad_norm=0.0,
        print_interval = None,
        save_dir = "train_net",
        criterion = "ISstd",
        last_only = False):
    
    if criterion not in ["ISstd", "KL1", "logstd", "KL_simp"]:
        raise ValueError("Invalid criterion")
    
    image_dir = None
    if save_dir is not None:
        checkpoint_dir = os.path.join(save_dir, "checkpoints")
        image_dir = os.path.join(save_dir, "images")
        for p in [save_dir, checkpoint_dir, image_dir]:
            if not os.path.exists(p):
                os.makedirs(p)
    
    optimiser = torch.optim.Adam(
        net.parameters(), lr = lr, weight_decay = wd)
    
    for i in range(trainer.iteration, trainer.iteration + nepochs):
        trainer.iteration = i
        
        if  trainer.eval_condition():
            results.eval_net(net, trainer, target_weight,
                             weight_log = weight_log, penalty = penalty,
                             criterion = criterion)
            if plot_fig is not None and image_dir is not None:
                plot_fig(net, trainer, results, image_dir)
                trainer.image_num += 1
            
            if not last_only:
                torch.save(
                    net, os.path.join(
                        checkpoint_dir, "net-{}.pt".format(trainer.iteration)))
            results.save(folder = save_dir)
            torch.save(trainer, os.path.join(save_dir, "trainer.pt"))
        
        net.train()
        optimiser.zero_grad()
        z = net.sample(trainer.bs)
        logpy, y, dgdz = net.logpy(z = z, return_zy = True, return_jac = True)
        weight = target_weight(y)
        if weight_log:
            logweight = weight
            weight = weight.exp()
        elif criterion != "ISstd":
            logweight = weight.log()
        
        if criterion == "ISstd":
            if penalty is not None:
                pen = penalty(y)
                ind = pen>=0
                pen_loss = torch.where(ind,pen*trainer.alpha, trainer.zero).mean()
                weight = torch.where(~ind, weight, trainer.zero)
                IS_loss = (weight/logpy.exp()).std(unbiased = True)
                loss = IS_loss + pen_loss
            else:
                IS_loss = (weight/logpy.exp()).std(unbiased = True)
                loss = IS_loss
        elif criterion == "KL1":
            if penalty is not None:
                pen = penalty(y)
                ind = pen>=0
                
                pen_loss=torch.where(ind,pen*trainer.alpha,trainer.zero).mean()
                KL1_loss = (logpy - logweight).mean()
                loss = KL1_loss + pen_loss
            else:
                KL1_loss = (logpy - logweight).mean()
                loss = KL1_loss
        elif criterion == "logstd":
            if penalty is not None:
                pen = penalty(y)
                ind = pen>=0
                
                pen_loss=torch.where(ind,pen*trainer.alpha,trainer.zero).mean()
                logstd_loss = (logpy - logweight).std(unbiased = True)
                loss = logstd_loss + pen_loss
            else:
                logstd_loss = (logpy - logweight).std(unbiased = True)
                loss = logstd_loss
        elif criterion == "KL_simp":
            if penalty is not None:
                pen = penalty(y)
                ind = pen>=0
                
                pen_loss=torch.where(ind,pen*trainer.alpha,trainer.zero).mean()
                KL_simp_loss = -(logweight + dgdz).mean()
                loss = KL_simp_loss + pen_loss
            else:
                KL_simp_loss = -(logweight + dgdz).mean()
                loss = KL_simp_loss
        
        loss.backward()
        if grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_norm)
        optimiser.step()
        
        if print_interval is not None:
            if i%print_interval==(print_interval-1):
                print(i+1,'loss: %.8f' % loss)
    
    trainer.iteration += 1
    results.eval_net(net, trainer, target_weight,
                     weight_log = weight_log, penalty = penalty,
                     criterion = criterion)
    if plot_fig is not None:
        plot_fig(net, trainer, results, image_dir = image_dir)
        trainer.image_num += 1
    torch.save(
        net, os.path.join(
            checkpoint_dir, "net-{}.pt".format(trainer.iteration)))
    results.save(folder = save_dir)
    torch.save(trainer, os.path.join(save_dir, "trainer.pt"))
        


def IS_scaled(logweight, logpy, trainer, ind = None, log_scale = 0,
              rescale = True):
    log_rat = logweight - logpy
    #log_IS_samples = torch.where(penalty(y)<0,log_rat,trainer.zero+1)
    
    if ind is not None:
        if ind.any() and rescale:
            log_scale = log_rat[ind].max()
        log_rat = log_rat - log_scale
        IS_samples = torch.where(ind,log_rat.exp(), trainer.zero)
    else:
        IS_samples = (log_rat - log_scale).exp()
    IS_m = IS_samples.mean()
    IS_s = IS_samples.std(unbiased = True)
    return IS_m, IS_s, log_scale, IS_samples







