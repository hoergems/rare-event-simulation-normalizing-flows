import torch
import torch.nn as nn


#------------------------------------------------------------------------------
# NORMALISING FLOW BASE CLASS


class NormFlow(nn.Module):
    """The base class for defining normalising flow modules
    
    Instance Attributes
    -------------------
    noise_distribution : torch.distributions.distribution.Distribution
        The noise or base distribution. Needs to have sample and log_prob
        methods. Typically noise_distribution is None when creating modules
        to be parts of a composite flow, and only defined in the final
        NormFlows instance.
    
    Methods Subclassed Should Override
    ----------------------------------
    g(z)
        Transforms noise samples z to y = g(z).
    f(y)
        Inverse of g. Transforms target values y to z = f(y).
    dgdz(z, return_y = False)
        Log of the absolute value of the determinant of the Jacobian matrix of
        g(z). Also returns y = g(z) if return_y is True.
    dfdy(y, return_z = False)
        Log of the absolute value of the determinant of the Jacobian matrix of
        f(y). Also returns z = f(y) if return_z is True.
    
    Methods Subclasses can Inherit
    ------------------------------
    forward(z)
        Returns y = g(z). This method is expected by nn.Module superclass.
    log_prob(z)
        Computes the log probability (density) at z.
    logpy(z = None, y = None, return_zy = False, return_jac = False)
        Computes the log probability (density) at y.
    sample(sample_shape = torch.Size([]))
        Samples from the noise/base distribution.
    sampley(sample_shape = torch.Size([]))
        Samples from the target distribution.
    """
    def __init__(self, noise_distribution = None):
        super().__init__()
        self.noise_distribution = noise_distribution
    
    def forward(self, z):
        y = self.g(z)
        return y
    
    def g(self, z):
        """Calculates the transformation from the noise variable Z, to Y.
        
        Y = g(Z), where Z is distributed according to self.noise_distribution.
        

        Parameters
        ----------
        z : torch.Tensor of shape [*, num dims]
            A tensor of samples from the noise distribution.

        Returns
        -------
        y : torch.Tensor of the same shape as z
            The transformed tensor.

        """
        raise NotImplementedError('Forward function has not been implemented.')
    
    def f(self, y):
        """Calculates the transformation from Y to the noise variable Z.
        
        Z = f(Y), where Z is distributed according to self.noise_distribution.
        f is the inverse function of g.

        Parameters
        ----------
        y : torch.Tensor of the same shape as z
            The transformed tensor.

        Returns
        -------
        z : torch.Tensor of shape [*, num dims]
            A tensor of samples from the noise distribution.

        """
        raise NotImplementedError('Inverse function has not been implemented.')
    
    def dgdz(self, z, return_y = False):
        """Calculates the log of the absolute determinant of the Jacobian.
        
        Derivative of g with respect to z.

        Parameters
        ----------
        z : torch.Tensor of shape [*, num dims]
            A tensor of samples from the noise distribution.
        return_y : boolean, optional
            Returns y when True. The default is False.

        Returns
        -------
        dgdz : torch.Tensor
            The log of the absolute value of the determinant of the jacobian
            matrix. The shape could be [*], [*,1] or [1].
        y : torch.Tensor of the same shape as z
            y = g(z). The transformed variable is returned when return_y is
            True.

        """
        raise NotImplementedError('Jacobian of g has not been implemented.')
    
    def dfdy(self, y, return_z = False):
        """Calculates the log of the absolute determinant of the Jacobian.
        
        Derivative of f with respect to y.

        Parameters
        ----------
        y : torch.Tensor of shape [*, num dims]
            A tensor of samples from the transformed distribution.
        return_z : boolean, optional
            Returns z when True. The default is False.

        Returns
        -------
        dfdy : torch.Tensor
            The log of the absolute value of the determinant of the jacobian
            matrix. The shape could be [*], [*,1] or [1].
        z : torch.Tensor of the same shape as z
            z = f(y). The transformed variable is returned when return_z is
            True.

        """
        raise NotImplementedError('Jacobian of f has not been implemented.')
    
    def log_prob(self, z):
        """Returns the log probability (density) evaluated at z.
        
        Mimics the torch.distributions.distribution.Distribution method of the
        same name.

        Parameters
        ----------
        z : torch.Tensor of shape [*, num dims]
            A tensor of samples from the noise distribution.

        Returns
        -------
        torch.Tensor of shape [*]
            The log of the probability (density).

        """
        if self.noise_distribution is None:
            raise NotImplementedError('Distribution has not been implemented.')
        return self.noise_distribution.log_prob(z)
    
    def logpy(self, z = None, y = None, return_zy = False, return_jac = False):
        """Returns the log probability (density) evaluated at y.
        
        dgdz is used when z is provided, otherwise dfdy is used.
        One of z and y needs to be provided.

        Parameters
        ----------
        z : torch.Tensor of shape [*, num dims], optional
            A tensor of samples from the noise distribution. The default is
            None.
        y : torch.Tensor of shape [*, num dims], optional
            A tensor of samples from the transformed distribution. The defualt
            is None.
        return_zy : boolean, optional
            Returns z or y when True. The default is False.
        return_jac : boolean, optional
            Returns dgdz or dfdy when True. The default is False.

        Returns
        -------
        logpy : torch.Tensor of shape [*]
            The log probability (density) of y.
        z : torch.Tensor of shape [*, num dims]
            Returned when z arg is None and return_zy is True.
        y : torch.Tensor of shape [*, num dims]
            Returned when y arg is None and return_zy is True.

        """
        if z is not None:
            dgdz = self.dgdz(z, return_y = return_zy)
            if return_zy:
                y = dgdz[1]
                logpy = self.log_prob(z) - dgdz[0]
                if return_jac:
                    return logpy, y, dgdz[0]
                else:
                    return logpy, y
            else:
                logpy = self.log_prob(z) - dgdz
                if return_jac:
                    return logpy, dgdz
                else:
                    return logpy
        elif y is not None:
            dfdy, z = self.dfdy(y, return_z = True)
            logpy = self.log_prob(z) + dfdy
            if return_zy:
                if return_jac:
                    return logpy, z, dfdy
                else:
                    return logpy, z
            else:
                if return_jac:
                    return logpy, dfdy
                else:
                    return logpy
        else:
            raise ValueError("z and y cannot both be None")
    
    def sample(self, sample_shape = torch.Size([])):
        """Samples from the noise distribution.
        

        Parameters
        ----------
        sample_shape : shape or int, optional
            The batch shape [*] of the sample. The default is torch.Size([]).

        Returns
        -------
        z : torch.Tensor of shape [*, num dims]
            The sample tensor from the noise distribution.

        """
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        z = self.noise_distribution.sample(sample_shape)
        return z
    
    def sampley(self, sample_shape = torch.Size([])):
        """Samples from the target distribution.
        
        Samples first from the noise distribution and then transforms to the
        target space.

        Parameters
        ----------
        sample_shape : shape or int, optional
            The batch shape [*] of the sample. The default is torch.Size([]).

        Returns
        -------
        y : torch.Tensor of shape [*, num dims]
            The sample tensor from the target distribution.

        """
        z = self.sample(sample_shape)
        y = self.g(z)
        return y
    
    def param_requires_grad(self, TF):
        for param in self.parameters():
            param.requires_grad = TF


# NORMALISING FLOW BASE CLASS
#------------------------------------------------------------------------------
# COMPOSITE NORMALISING FLOWS CLASS


class NormFlows(NormFlow):
    """The base class for defining composite normalising flows
    
    Instance Attributes
    -------------------
    flows : torch.nn.Sequential
        A sequential Module of the 'NormFlow' components. This can be a list
        or tuple when input into the constructor.
    
    """
    def __init__(self, flows, noise_distribution = None):
        if len(flows) < 1:
            raise ValueError("flows needs to have length of at least 1")
        super().__init__(noise_distribution = noise_distribution)
        if not isinstance(flows, nn.Sequential):
            self.flows = nn.Sequential(*flows)
        else:
            self.flows = flows
    
    def g(self, z, params = None):
        y = z
        if params is not None:
            for p, flow in zip(params, self.flows):
                y = flow.g(y, params = p)
        else:
            for flow in self.flows:
                y = flow.g(y)
        return y
    
    def f(self, y, params = None):
        z = y
        if params is not None:
            for p, flow in zip(params[::-1], self.flows[::-1]):
                z = flow.f(z, params = p)
        else:
            for flow in self.flows[::-1]:
                z = flow.f(z)
        return z
    
    def dgdz(self, z, return_y = False, params = None):
        dgdz = 0
        y = z
        if params is not None:
            for p, flow in zip(params, self.flows):
                d, y = flow.dgdz(y, return_y = True, params = p)
                dgdz = dgdz + d
        else:
            for flow in self.flows:
                d, y = flow.dgdz(y, return_y = True)
                dgdz = dgdz + d
        if return_y:
            return dgdz, y
        else:
            return dgdz
    
    def dfdy(self, y, return_z = False, params = None):
        dfdy = 0
        z = y
        if params is not None:
            for p, flow in zip(params[::-1], self.flows[::-1]):
                d, z = flow.dfdy(z, return_z = True, params = p)
                dfdy = dfdy + d
        else:
            for flow in self.flows[::-1]:
                d, z = flow.dfdy(z, return_z = True)
                dfdy = dfdy + d
        if return_z:
            return dfdy, z
        else:
            return dfdy


# COMPOSITE NORMALISING FLOWS CLASS
#------------------------------------------------------------------------------
# MULTILAYER PERCEPTRON CLASS


class MLP(nn.Module):
    """Multi-Layer Perceptron class.
    
    """
    def __init__(self,
                 in_shape,       # The input shape
                 out_shape,      # The output shape
                 HLsizes = None, # A list of hidden layer sizes
                 HLact = None,   # Hidden layer activation function
                 outact = None,  # Output layer activation function
                 parameter_max_bound = 0.1 # Initialising parameter bound
                ):
        """
        Parameters
        ----------
        in_shape : list
            The shape of the input tensor, excluding batch dimensions.
        out_shape : TYPE
            The shape of the output tensor, excluding batch dimensions.
        HLsizes : list, optional
            The sizes of each hidden layer. The default is an empty list (no
            hidden layers).
        HLact : function, optional
            The activation function used at each hidden layer. The default is
            leaky ReLU.
        outact : function, optional
            The activation function of the output layer. The default is the
            identity function.
        parameter_max_bound : float, optional
            The maximum absolute value the parameters can be initialised to be.
            The default is 0.1.

        """
        super().__init__()
        if HLsizes is None:
            HLsizes = []
        if HLact is None:
            HLact = nn.LeakyReLU()
        if outact is None:
            outact = nn.Identity()
        if isinstance(in_shape, int):
            in_shape = [in_shape]
        if isinstance(out_shape, int):
            out_shape = [out_shape]
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.in_dims = len(in_shape)
        self.out_dims = len(out_shape)
        self.HLsizes = HLsizes
        self.HLact = HLact
        self.outact = outact
        self.parameter_max_bound = parameter_max_bound
        
        self.insize = torch.prod(torch.tensor(in_shape)).item()
        self.outsize = torch.prod(torch.tensor(out_shape)).item()
        sizes = [self.insize] + HLsizes + [self.outsize]
        n_layer = len(sizes) - 1
        self.linears = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i+1]) for i in range(n_layer)])
        
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.parameter_max_bound is None:
            for L in self.linears:
                L.reset_parameters()
        else:
            for L in self.linears:
                bound = torch.sqrt(torch.tensor(1/L.in_features))
                bound = bound.clamp(max = self.parameter_max_bound).item()
                L.weight.data.uniform_(-bound, bound)
                if L.bias is not None:
                    L.bias.data.uniform_(-bound, bound)
    
    def forward(self, x):
        s = x.shape[:-self.in_dims]
        x = x.view(*s,self.insize)
        for L in self.linears[:-1]:
            x = self.HLact(L(x))
        x = self.linears[-1](x)
        x = self.outact(x)
        x = x.view(*s,*self.out_shape)
        return x
    
    def param_requires_grad(self, TF):
        for param in self.parameters():
            param.requires_grad = TF


# MULTILAYER PERCEPTRON CLASS
#------------------------------------------------------------------------------
# AUTOREGRESSIVE FLOW CLASSES


class InverseAutoRegressiveFlow(NormFlow):
    """
    WIP
    """
    def __init__(self, D, noise_distribution = None,
                 conditioner = None, coupler = None,
                 yi = None):
        super().__init__(noise_distribution = noise_distribution)
        
        raise NotImplementedError('InverseAutoRegressiveFlow is a WIP.')
        
        if conditioner is None:
            conditioner = MLPConditioner([D,1],[D,5])
        if coupler is None:
            coupler = NonlinearSquaredFlow(D = D, include_params = False)
        self.D = D
        self.conditioner = conditioner
        self.coupler = coupler
        self.yi = yi
    
    def conditioner_default(self, y):
        # y shape should be [..., D, num steps]
        # default conditioner only uses previous time step
        y = y[...,-1]
    
    def g_step(self, z, y0):
        theta = self.conditioner(y0)
        y = self.coupler.g(z, params = theta)
        return y
    
    def f_step(self, y, y0):
        theta = self.conditioner(y0)
        z = self.coupler.f(y, params = theta)
        return z
    
    def dgdz_step(self, z, y0, return_y = False):
        theta = self.conditioner(y0)
        if return_y:
            dgdz, y = self.coupler.dgdz(z, params = theta, return_y = True)
            return dgdz, y
        else:
            dgdz = self.coupler.dgdz(z, params = theta)
            return dgdz
    
    def dfdy_step(self, y, y0, return_z = False):
        theta = self.conditioner(y0)
        if return_z:
            dfdy, z = self.coupler.dfdy(y, params = theta, return_z = True)
            return dfdy, z
        else:
            dfdy = self.coupler.dfdy(y, params = theta)
            return dfdy
    
    def g(self, z, y0 = None):
        if y0 is None:
            y0 = self.yi
        batch_shape = z.shape[:-1]
        z = z.view(*batch_shape, self.D, -1)
        steps = z.shape[-1]
        y = torch.zeros_like(z)
        y0 = self.g_step(z[...,0], y0.unsqueeze(-1))
        y[...,0] = y0
        for i in range(1,steps):
            y0 = self.g_step(z[...,i], y[...,:i])
            y[...,i] = y0
        y = y.view(*batch_shape, -1)
        return y


class MLPConditioner(MLP):
    """
    Modifies foward method of MLP class to make the last dimension match
    the expected input. It does this be either repeating the first part of
    the input tensor (a kind of padding) if it is too small, or by choosing
    the last part of the input tensor if it is too big.
    """
    def forward(self, x):
        num_steps = self.in_shape[-1]
        xsteps = x.shape[-1]
        # pad tensor by repeating x[...,[0]]
        if xsteps < num_steps:
            x = torch.cat((
                x[...,[0]].repeat((x.ndim-1)*[1] + [num_steps - xsteps]),
                x
                ), dim = -1)
        
        x = x[...,-num_steps:]
        x = super().forward(x)
        return x


# AUTOREGRESSIVE FLOW CLASSES
#------------------------------------------------------------------------------
# NORMALISING FLOW CLASSES


class CouplingFlow(NormFlow):
    """
    y1 = z1
    y2 = h(z2, theta(z1))
    
    z1 = y1
    z2 = h^-1(y2, theta(y1))
    
    h is the coupling function and must be bijective
    theta computed from z1/y1 using the conditioner function.
    
    jacobian is jacobian of h
    
    By default the coupler is a NonlinearSquaredFlowSeq object, and the
    conditioner is an MLP object.
    """
    def __init__(self, D, C = None, noise_distribution = None,
                 conditioner = None, coupler = None,
                 normalise = False, fix_zero = False, N = 1, **mlpkwargs):
        super().__init__(noise_distribution = noise_distribution)
        if C is None:
            C = D//2
        self.check_dims(D, C)
        if conditioner is None:
            n = 3 if normalise else (4 if fix_zero else 5)
            conditioner = MLP(C,[D-C,n,N], **mlpkwargs)
        if coupler is None:
            coupler = NonlinearSquaredFlowSeq(
                N = N,
                D = (D-C), include_params = False,
                normalise = normalise, fix_zero = fix_zero)
        self.N = N
        self.D = D
        self.C = C # number of dimensions used to input the conditioner
        self.conditioner = conditioner
        self.coupler = coupler
    
    def check_dims(self, D, C):
        if D < 2:
            raise ValueError("D must be at least 2")
        if (C >= D) or (C < 1):
            raise ValueError("C must be at least 1 and less than D")
    
    def g(self, z):
        z1 = z[..., :self.C]
        z2 = z[..., self.C:]
        theta = self.conditioner(z1)
        theta = [theta[...,i] for i in range(self.N)]
        y2 = self.coupler.g(z2, params = theta)
        y = torch.cat((z1, y2), dim = -1)
        return y
    
    def f(self, y):
        y1 = y[..., :self.C]
        y2 = y[..., self.C:]
        theta = self.conditioner(y1)
        theta = [theta[...,i] for i in range(self.N)]
        z2 = self.coupler.f(y2, params = theta)
        z = torch.cat((y1, z2), dim = -1)
        return z
    
    def dgdz(self, z, return_y = False):
        z1 = z[..., :self.C]
        z2 = z[..., self.C:]
        theta = self.conditioner(z1)
        theta = [theta[...,i] for i in range(self.N)]
        if return_y:
            dgdz, y2 = self.coupler.dgdz(z2, params = theta, return_y = True)
            y = torch.cat((z1, y2), dim = -1)
            return dgdz, y
        else:
            dgdz = self.coupler.dgdz(z2, params = theta)
            return dgdz
    
    def dfdy(self, y, return_z = False):
        y1 = y[..., :self.C]
        y2 = y[..., self.C:]
        theta = self.conditioner(y1)
        theta = [theta[...,i] for i in range(self.N)]
        if return_z:
            dfdy, z2 = self.coupler.dfdy(y2, params = theta, return_z = True)
            z = torch.cat((y1, z2), dim = -1)
            return dfdy, z
        else:
            dfdy = self.coupler.dfdy(y2, params = theta)
            return dfdy


class LinearFlow(NormFlow):
    """Linear Flow
    
    let z and y be a column vectors with dimensionality D.
    b and W contain the parameters. b is a vector of size D and W is a DxD
    matrix. Invertibility is not guaranteed under the current implimentation.
    
    y = g(z) = Wz + b
    z = f(y) = W^-1(y-b)
    dgdz = log(abs(det(W)))
    dfdy = log(abs(det(W^-1)))
    """
    def __init__(self, D, bias = True, noise_distribution = None):
        super().__init__(noise_distribution = noise_distribution)
        self.D = D
        self.linear = nn.Linear(D, D, bias = bias)
        self.bias = bias
        self.reset_parameters()
    
    def reset_parameters(self):
        self.linear.reset_parameters()
        I = torch.eye(self.D)
        with torch.no_grad():
            self.linear.weight = nn.Parameter(self.linear.weight + I)
    
    def g(self, z):
        y = self.linear(z)
        return y
    
    def f(self, y):
        init_shape = y.shape
        if self.bias:
            y = y - self.linear.bias
        tr = y.dim() > 1
        if tr:
            y = y.transpose(-2,-1)
        
        w = self.linear.weight
        z = torch.linalg.solve(w, y)
        if tr:
            z = z.transpose(-2,-1)
        z = z.view(init_shape)
        return z
    
    def dgdz(self, z, return_y = False):
        dgdz = self.linear.weight.det().abs().log()
        if return_y:
            y = self.g(z)
            return dgdz, y
        else:
            return dgdz
    
    def dfdy(self, y, return_z = False):
        dfdy = -self.dgdz(y)
        if return_z:
            z = self.f(y)
            return dfdy, z
        else:
            return dfdy


class ShiftFlow(NormFlow):
    """
    
    y = g(z) = z + b
    
    """
    def __init__(self, D = 1, noise_distribution = None):
        super().__init__(noise_distribution = noise_distribution)
        self.b = nn.Parameter(torch.Tensor(D), requires_grad = True)
        self.reset_parameters()
        self.D = D
    
    def reset_parameters(self):
        self.b.data.uniform_(-0.1, 0.1)
    
    def g(self, z):
        return z + self.b
    
    def f(self, y):
        return y - self.b
    
    def dgdz(self, z, return_y = False):
        if return_y:
            return 0, self.g(z)
        else:
            return 0
    
    def dfdy(self, y, return_z = False):
        if return_z:
            return 0, self.f(y)
        else:
            return 0


class DiagonalFlow(NormFlow):
    """
    
    y = g(z) = A z
    
    Where A is a scalar or diagonal matrix.
    
    """
    def __init__(self, D = 1, noise_distribution = None, monotone = False):
        super().__init__(noise_distribution = noise_distribution)
        self.A = nn.Parameter(torch.Tensor(D), requires_grad = True)
        self.reset_parameters()
        self.D = D
        self.monotone = monotone
    
    def reset_parameters(self):
        self.A.data.uniform_(-0.1, 0.1)
    
    def g(self, z):
        A = self.A
        if self.monotone:
            A = A.exp()
        return z*A
    
    def f(self, y):
        A = self.A
        if self.monotone:
            A = A.exp()
        return y/A
    
    def dgdz(self, z, return_y = False):
        if self.monotone:
            d = self.A.sum()
        else:
            d = self.A.abs().log().sum()
        if return_y:
            return d, self.g(z)
        else:
            return d
    
    def dfdy(self, y, return_z = False):
        if self.monotone:
            d = -self.A.sum()
        else:
            d = -self.A.abs().log().sum()
        if return_z:
            return d, self.f(y)
        else:
            return d


class Rank1LinearFlow(NormFlow):
    """
    Planar flow with identity h and zero bias.
    
    y = g(z) = Az
    
    where A = I + uv^T, u and v are vectors
    """
    def __init__(self, D, noise_distribution = None):
        super().__init__(noise_distribution = noise_distribution)
        self.u = nn.Parameter(torch.Tensor(D), requires_grad = True)
        self.v = nn.Parameter(torch.Tensor(D), requires_grad = True)
        self.reset_parameters()
        self.D = D
    
    @property
    def A(self):
        A = torch.eye(self.D) + torch.outer(self.u, self.v)
        return A
    
    @property
    def Adet(self):
        return 1 + torch.sum(self.u*self.v, keepdim = True)
    
    @property
    def Ainv(self):
        A = torch.eye(self.D) - torch.outer(self.u, self.v)/self.Adet
        return A
    
    def reset_parameters(self):
        self.u.data.uniform_(-0.1, 0.1)
        self.v.data.uniform_(-0.1, 0.1)
    
    def g(self, z):
        return z.matmul(self.A.T)
    
    def f(self, y):
        return y.matmul(self.Ainv.T)
    
    def dgdz(self, z, return_y = False):
        d = self.Adet.abs().log()
        if return_y:
            return d, self.g(z)
        else:
            return d
    
    def dfdy(self, y, return_z = False):
        d = -self.Adet.abs().log()
        if return_z:
            return d, self.f(y)
        else:
            return d


class TriangularFlow(NormFlow):
    """
    
    y = g(z) = A z
    
    Where A is a triangular matrix.
    
    """
    def __init__(self, D, noise_distribution = None):
        super().__init__(noise_distribution = noise_distribution)
        self.tri = nn.Parameter(torch.Tensor(D*(D+1)//2), requires_grad = True)
        self.reset_parameters()
        self.D = D
        self.ind = torch.triu_indices(D, D)
        #self.cache = torch.zeros(D, D)
    
    @property
    def A(self):
        A = torch.zeros(self.D, self.D)
        A[self.ind[0],self.ind[1]] = self.tri
        return A
    
    def reset_parameters(self):
        self.tri.data.uniform_(-0.1, 0.1)
    
    def g(self, z):
        return z.matmul(self.A.T)
    
    def f(self, y):
        tr = y.dim() > 1
        if tr:
            y = y.transpose(-2,-1)
        
        z = torch.linalg.solve(self.A, y)
        if tr:
            z = z.transpose(-2,-1)
        z = z.view(y.shape)
        return z
    
    def dgdz(self, z, return_y = False):
        d = self.A.diag().abs().log().sum()
        if return_y:
            return d, self.g(z)
        else:
            return d
    
    def dfdy(self, y, return_z = False):
        d = -self.A.diag().abs().log().sum()
        if return_z:
            return d, self.f(y)
        else:
            return d


class SylvesterFlow(NormFlow):
    """Sylvester Flow
    
    Invertibility is not guaranteed under the current implimentation.
    
    y = g(z) = z + U h(W^T z + b)
    dgdz = log(abs(det(I_M + diag(h'(W^T z + b))WU^T)))
    
    U, W and b contain the learnable paramters.
    U and W are matrices of size DxM where M<=D. b is a vecor of length M.
    
    """
    def __init__(self, D, M, h = None, dh = None, noise_distribution = None):
        super().__init__(noise_distribution = noise_distribution)
        if h is None:
            h = identity
            dh = torch.ones_like
        self.M = M
        self.h = h
        self.dh = dh
        self.inner = nn.Linear(D, M)
        self.outer = nn.Linear(M, D, bias = False)
    
    def reset_parameters(self):
        self.inner.reset_parameters()
        self.outer.reset_parameters()
    
    def g(self, z):
        y = self.inner(z)
        y = self.h(y)
        y = z + self.outer(y)
        return y
    
    def dgdz(self, z, return_y = False):
        W = self.inner.weight.mm(self.outer.weight)
        inner = self.inner(z)
        dgdz = self.dh(inner).unsqueeze(-2)*W + torch.eye(self.M)
        dgdz = dgdz.det().log()
        
        if return_y:
            y = z + self.outer(self.h(inner))
            return dgdz, y
        else:
            return dgdz


class ElementwiseFlow(NormFlow):
    """Elementwise FLow
    
    
    y = g(z) = h(z)
    z = f(y) = h^-1(y)
    dgdz = log(abs(prod(h'(z))))
    dfdy = log(abs(prod(h^-1'(z))))
    h is an invertible transformation with diagonal Jacobian matrix.
    """
    def __init__(self, function = None, inverse = None, derivative = None,
                 inv_derivative = None):
        super().__init__()
        if function is None:
            function = identity
            inverse = identity
            derivative = torch.ones_like
            inv_derivative = torch.ones_like
        
        self.function = function
        self.inverse = inverse
        self.derivative = derivative
        self.inv_derivative = inv_derivative
    
    def g(self, z):
        return self.function(z)
    
    def f(self, y):
        return self.inverse(y)
    
    def dgdz(self, z, return_y = False):
        dgdz = self.derivative(z).abs().log().sum(-1)
        if return_y:
            y = self.g(z)
            return dgdz, y
        else:
            return dgdz
    
    def dfdy(self, y, return_z = False):
        inv_cond = self.inv_derivative is not None
        if inv_cond:
            dfdy = self.inv_derivative(y).abs().log().sum(-1)
        else:
            z = self.f(y)
            dfdy = -self.dgdz(z)
        if return_z:
            if inv_cond:
                z = self.f(y)
            return dfdy, z
        else:
            return dfdy




class ElementwiseCubicFlow(ElementwiseFlow):
    """Elementwise Flow where h(z) = z + z^3
    """
    def __init__(self):
        super().__init__(
            function = cubic,
            inverse = inv_cubic,
            derivative = der_cubic,
            inv_derivative = der_inv_cubic)


class ElementwiseInvCubicFlow(ElementwiseFlow):
    """Elementwise Flow where h^-1(y) = y + y^3
    """
    def __init__(self):
        super().__init__(
            function = inv_cubic,
            inverse = cubic,
            derivative = der_inv_cubic,
            inv_derivative = der_cubic)


class ElementwiseExpFlow(ElementwiseFlow):
    """Elementwise Flow where h(z) = exp(z)
    """
    def __init__(self):
        super().__init__(
            function = torch.exp,
            inverse = torch.log,
            derivative = torch.exp,
            inv_derivative = self.one_over_y)
    
    def one_over_y(self, y):
        return 1/y


class ElementwiseSigmoidFlow(ElementwiseFlow):
    """Elementwise Flow where h(z) = a/(1+exp(-z))+b
    a = 1 and b = 0 by default (giving a range of (0,1)).
    
    Optionally, y can be clamped between eps and 1-eps for stability.
    """
    def __init__(self, eps = None, bounds = (0,1)):
        super().__init__(
            function = self.clamped_sigmoid,
            inverse = self.clamped_logit,
            derivative = self.sigmoid_derivative,
            inv_derivative = self.logit_derivative)
        self.eps = eps
        self.bounds = bounds
        self.a = bounds[1] - bounds[0]
        self.b = bounds[0]
    
    def clamped_sigmoid(self, z):
        z = torch.sigmoid(z)
        if self.eps is not None:
            z = z.clamp(self.eps, 1-self.eps)
        z = self.a*z + self.b
        return z
    
    def clamped_logit(self, y):
        y = (y-self.b)/self.a
        return torch.logit(y, eps = self.eps)
    
    def sigmoid_derivative(self, z):
        s = torch.sigmoid(z)
        if self.eps is not None:
            s = s.clamp(self.eps, 1-self.eps)
        return s*(1-s)*self.a
    
    def logit_derivative(self, y):
        y = (y-self.b)/self.a
        if self.eps is not None:
            y = y.clamp(self.eps, 1-self.eps)
        return (1/y+1/(1-y))/self.a


class NonlinearSquaredFlow(NormFlow):
    """
    
    y = g(z) = a*z+b + c/(1+(d*z+h)**2)
    
    The parameters a, b, c, d, & h are learnable.
    
    The jacobian is diagonal.
    
    Ziegler and Rush, Latent Normalizing Flows for Discrete Sequences, 2019
    
    Optionally, if normalise is set to True, then there are only 3 required
    parameters since a and b will depend on c, d & h
    such that g(0) = 0 and g(1) = 1. This ensures that if z in [0,1] then
    y is also in [0,1] (and vice versa).
    
    Optionally, if fix_zero is set to True, then b is set to -c/(1+h**2) so
    that g(0) = 0. Since a is transformed to be positive, then the sign of z
    will equal the sign of y. Also, there are only 4 required paramers.
    """
    def __init__(self, D = 1, noise_distribution = None, alpha = 0.95,
                 include_params = True, normalise = False, fix_zero = False):
        super().__init__(noise_distribution = noise_distribution)
        self.include_params = include_params
        self.normalise = normalise
        self.fix_zero = fix_zero
        self.alpha = alpha
        self.factor = 8*3**(1/2)/9 # used to transform c
        if include_params:
            n = 3 if normalise else (4 if fix_zero else 5)
            self.params = nn.Parameter(torch.Tensor(D,n), requires_grad = True)
            self.reset_parameters()
    
    def reset_parameters(self):
        if self.include_params:
            self.params.data.uniform_(-0.1, 0.1)
    
    def transformed_params(self, params = None):
        if params is None:
            params = self.params
        if self.normalise:
            d = params[...,1].exp()
            h = params[...,2]
            t = self.factor*self.alpha*params[...,0].tanh()/d
            oneh2 = 1/(1+h**2)
            a = 1/(1-t*oneh2+t/(1+(d+h)**2))
            c = t*a
            b = -c*oneh2
        elif self.fix_zero:
            a = params[...,0].exp()
            d = params[...,2].exp()
            h = params[...,3]
            c = self.factor*self.alpha*a*params[...,1].tanh()/d
            b = -c/(1+h**2)
        else:
            a = params[...,0].exp()
            b = params[...,1]
            d = params[...,3].exp()
            h = params[...,4]
            c = self.factor*self.alpha*a*params[...,2].tanh()/d
        return a,b,c,d,h
    
    def g(self, z, params = None, coeffs = None):
        if coeffs is None:
            coeffs = self.transformed_params(params = params)
        a,b,c,d,h = coeffs
        return a*z+b + c/(1+(d*z+h)**2)
    
    def f(self, y, params = None, coeffs = None):
        if coeffs is None:
            coeffs = self.transformed_params(params = params)
        a,b,c,d,h = coeffs
        A = (b-y)*d/a-h
        B = A + c*d/a
        q = -(A/3)**3 + (A-3*B)/6
        k = (q**2 + (1/3-(A/3)**2)**3)**(1/2)
        K1 = q + k
        K2 = q - k
        K3 = K1.sign()*K1.abs().pow(1/3) + K2.sign()*K2.abs().pow(1/3) - A/3
        return (K3 - h)/d
    
    def dgdz(self, z, return_y = False, params = None, coeffs = None):
        if coeffs is None:
            coeffs = self.transformed_params(params = params)
        a,b,c,d,h = coeffs
        derivative = a - 2*c*d*(d*z+h)/(1+(d*z+h)**2)**2
        dgdz = derivative.abs().log().sum(-1)
        if return_y:
            y = self.g(z, coeffs = coeffs)
            return dgdz, y
        else:
            return dgdz
    
    def dfdy(self, y, return_z = False, params = None, coeffs = None):
        if coeffs is None:
            coeffs = self.transformed_params(params = params)
        z = self.f(y, coeffs = coeffs)
        dfdy = -self.dgdz(z, coeffs = coeffs)
        if return_z:
            return dfdy, z
        else:
            return dfdy


class NonlinearSquaredFlowSeq(NormFlows):
    """A sequence of instances of NonlinearSquaredFlow.
    """
    def __init__(self, N = 2, noise_distribution = None, **kwargs):
        flows = [NonlinearSquaredFlow(**kwargs) for i in range(N)]
        self.N = N
        super().__init__(flows, noise_distribution = noise_distribution)



class PermutationFlow(NormFlow):
    """Permutes elements
    y = g(z) = z[perm]
    z = f(y) = y[perm^-1]
    dgdz = 0
    dfdy = 0
    
    Where perm is a permutation vector (e.g. [2,0,1,3])
    and perm^-1 is the inverse permutation vector (e.g. [1,2,0,3])
    """
    def __init__(self, perm = None):
        super().__init__()
        if perm is None:
            perm = torch.tensor([], dtype = int)
        if not isinstance(perm, torch.Tensor):
            perm = torch.tensor(perm, dtype = int)
        self.perm = perm
        self.invperm = self.perm.argsort()
    
    def g(self, z):
        return z[...,self.perm]
    
    def f(self, y):
        return y[...,self.invperm]
    
    def dgdz(self, z, return_y = False):
        if return_y:
            return 0, self.g(z)
        else:
            return 0
    
    def dfdy(self, y, return_z = False):
        if return_z:
            return 0, self.f(y)
        else:
            return 0









# NORMALISING FLOW CLASSES
#------------------------------------------------------------------------------
# FUNCTIONS


def identity(x):
    return x


def cubic(x):
    return x + x**3

def inv_cubic(y):
    k = (((3*(27*y**2 + 4))**(1/2) + 9*y)*3/2)**(1/3)
    return k/3 - 1/k

def der_cubic(x):
    return 1 + 3*x**2

def der_inv_cubic(y):
    return 1/der_cubic(inv_cubic(y))


# FUNCTIONS
#------------------------------------------------------------------------------
#








