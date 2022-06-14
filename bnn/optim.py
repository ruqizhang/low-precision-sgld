import torch
from torch.optim import Optimizer, SGD, Adam
from qtorch import BlockFloatingPoint, FixedPoint, FloatingPoint
from qtorch.quant import *
import numpy as np
__all__ = ["OptimLP"]


class OptimLP(Optimizer):
    """
    A low-precision optimizer wrapper that handles weight, gradient, accumulator quantization.

    Args:
        - :attr: `optim`: underlying optimizer to use
        - :attr: `weight_quant`: a weight quantization function which takes a pytorch tensor and returns a tensor. If None, does not quantize weight.
        - :attr: `grad_quant`: a gradient quantization function which takes a pytorch tensor and returns a tensor. If None, does not quantize weight.
        - :attr: `grad_scaling`: float, scaling factor before apply gradient quantization.
        - :attr: `momentum_quant`: a momentum quantization function which takes a pytorch tensor and returns a tensor.
                                   If None, does not quantize weight.
        - :attr: `acc_quant`: a accumulator quantization function which takes
                              a pytorch tensor and returns a tensor. If not None, a
                              OptimLP object would create memory copies of model parameters that serve as
                              gradient accumulators. If None, does not use gradient accumulators.

    Example:
        >>> weight_q = quantizer(...) # define weight quantization
        >>> optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer = OptimLP(optiimizer, weight_quant=weight_q)
    """

    def __init__(
        self,
        optim,
        weight_quant=None,
        grad_scaling=1.0,
        grad_quant=None,
        acc_quant=None,
        noise=False,
        temperature=1.0,
        datasize=None,
        WL=8,
        FL=8,
        EXP=8,
        MAN=7,
        quant_type='naive',
        number_type='fixed'
    ):
        assert isinstance(optim, SGD) or isinstance(optim, Adam)
        super(OptimLP, self).__init__(
            optim.param_groups, optim.defaults
        )  # place holder

        # python dictionary does not copy by default
        self.param_groups = optim.param_groups
        self.optim = optim

        assert grad_scaling > 0, "gradient scaling must be positive"
        self.grad_scaling = grad_scaling

        self.weight_quant = weight_quant
        self.grad_quant = grad_quant
        self.acc_quant = acc_quant

        if self.acc_quant != None:
            self.weight_acc = {}
            for group in self.param_groups:
                for p in group["params"]:
                    self.weight_acc[p] = p.detach().clone()
        self.noise = noise
        self.temperature = temperature
        self.datasize = datasize
        self.quant_type = quant_type
        self.WL = WL
        self.FL = FL
        self.EXP = EXP
        self.MAN = MAN
        self.number_type = number_type
        self.ebit = 8
        if quant_type=='vc':
            self.D = 1./(2**FL)
            self.var_fix = self.D**2/4.
            if self.number_type == 'fixed':
                number = FixedPoint(wl=WL, fl=FL)
            elif self.number_type == 'block':
                number = BlockFloatingPoint(WL,dim=0)   
            elif self.number_type == 'float':
                number = FloatingPoint(EXP, MAN)                       
            self.quant_s = quantizer(
                forward_number=number, forward_rounding="stochastic"
            )
            self.quant_n = quantizer(
                forward_number=number, forward_rounding="nearest"
            )
    def step(self, closure=None):
        """
        Performs one step of optimization with the underlying optimizer.
        Quantizes gradient and momentum before stepping. Quantizes gradient accumulator and weight after stepping.
        """
        loss = None
        # quantize gradient
        if not self.grad_quant is None:
            for group in self.param_groups:
                for p in group["params"]:
                    p.grad.data = self.grad_quant(p.grad.data * self.grad_scaling)
    
        # switch acc into weight before stepping
        if not self.acc_quant is None:
            for group in self.param_groups:
                for p in group["params"]:
                    p.data = self.weight_acc[p].data
    
        self.update_params()

        # switch weight into acc after stepping and quantize
        if not self.acc_quant is None:
            for group in self.param_groups:
                for p in group["params"]:
                    if self.acc_quant == "full":
                        self.weight_acc[p].data = p.data
                    else:
                        p.data = self.weight_acc[p].data = self.acc_quant(p.data).data
                    

        # quantize weight from acc
        if (not self.weight_quant is None) and (self.quant_type=='naive'):
            for group in self.param_groups:
                for p in group["params"]:
                    p.data = self.weight_quant(p.data).data
        
        return loss

    def update_params(self):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for p in group['params']:
                param_state = self.optim.state[p]
                d_p = p.grad.data
                d_p.add_(p.data, alpha=weight_decay)
                p.data.add_(d_p, alpha=-group['lr'])
                if self.noise:
                    if self.quant_type == 'vc':
                        var = 2.0*group['lr']*self.temperature/self.datasize
                        if self.number_type == 'block':
                            self.FL = self.compute_fl(p.data)
                            self.D = 1./(2**self.FL)
                            self.var_fix = self.D**2/4. 
                            p.data = self.fp_Q_vc(p.data,var) 
                        elif self.number_type == 'float':
                            self.FL = self.compute_fl_float(p.data)
                            self.D = 1./(2**self.FL)
                            self.var_fix = self.D**2/4. 
                            p.data = self.fp_Q_vc(p.data,var) 
                        else:                                                       
                            p.data = self.Q_vc(p.data,var)                       
                    else:
                        eps = torch.randn(p.size(),device='cuda')
                        noise = (2.0*group['lr']*self.temperature/self.datasize)**.5*eps
                        p.data.add_(noise)

    def quant_n_hf(self, mu):
        return mu.half()
             
    def compute_fl_float(self, mu):
        max_entry = torch.abs(mu)
        max_exponent = torch.floor(torch.log2(max_entry))
        max_exponent = torch.clamp(max_exponent, -2**(self.EXP-1), 2**(self.EXP-1)-1)
        return self.MAN-max_exponent

    def compute_fl(self, mu):
        max_entry = torch.max(torch.abs(mu.view(mu.size(0), -1)), 1)[0]
        max_exponent = torch.floor(torch.log2(max_entry))
        max_exponent = torch.clamp(max_exponent, -2**(self.ebit-1), 2**(self.ebit-1)-1)
        max_exponent = max_exponent.view([mu.size(0)]+[1 for _ in range(mu.dim()-1)])
        max_exponent = max_exponent.expand([-1]+[mu.size(i) for i in range(1,mu.dim())])
        return self.WL-2-max_exponent

    def Q_vc(self, mu, var):
        if var>self.var_fix:
            x = mu + (var-self.var_fix)**.5*torch.randn(mu.size(),device='cuda')
            quant_x = self.quant_n(x)
            residual = x - quant_x
            theta = quant_x + torch.sign(residual)*self.sample_mu(torch.abs(residual))
        else:
            quant_mu = self.quant_s(mu)
            residual = mu - quant_mu
            p1 = torch.abs(residual)/self.D
            var_s = (1.-p1)*residual**2+p1*(-residual+torch.sign(residual)*self.D)**2
            v = var-var_s
            v[(v<0).nonzero(as_tuple=True)]=0
            theta = quant_mu + self.sample(v)

        theta = torch.clamp(theta, min=-2**(self.WL-self.FL-1), max=2**(self.WL-self.FL-1)-2**(-self.FL))
        return theta

    def fp_Q_vc(self, mu, var):
        FL0 = self.FL.detach()
        var = torch.zeros(1,device='cuda')+var
        var = var.expand(self.var_fix.size())
        ind = (var<=self.var_fix).nonzero(as_tuple=True)

        x = mu + (var-self.var_fix)**.5*torch.randn(mu.size(),device='cuda')
        quant_x = self.quant_n(x)
        if self.number_type == 'block':
            self.FL = self.compute_fl(x)
            self.D = 1./(2**self.FL)
        elif self.number_type == 'float':
            self.FL = self.compute_fl_float(x)
            self.D = 1./(2**self.FL)
        residual = x - quant_x
        theta = quant_x + torch.sign(residual)*self.fp_sample_mu(torch.abs(residual))

        quant_mu = self.quant_s(mu)
        residual = mu - quant_mu
        p1 = torch.abs(residual)/self.D
        var_s = (1.-p1)*residual**2+p1*(-residual+torch.sign(residual)*self.D)**2
        v = var-var_s
        v[(v<0).nonzero(as_tuple=True)]=0
        theta1 = quant_mu + self.fp_sample(v)

        theta[ind] = theta1[ind]
        self.FL[ind] = FL0[ind]
        if self.number_type == 'float':
            pass
        else:
            theta = torch.clamp(theta, min=-2**(self.WL-self.FL-1), max=2**(self.WL-self.FL-1)-2**(-self.FL))
        return theta

    def sample(self,var):
        p1 = var/(2*self.D**2)
        u = torch.rand(var.size(),device='cuda')
        s = torch.zeros(var.size(),device='cuda')
        s[(u<p1).nonzero(as_tuple=True)] = self.D 
        u[(u<p1).nonzero(as_tuple=True)] = 10.
        s[(u<(2*p1)).nonzero(as_tuple=True)] = -self.D 
        return s

    def sample_mu(self,mu):
        p1 = (self.var_fix+mu**2+mu*self.D)/(2*self.D**2)
        p2 = (self.var_fix+mu**2-mu*self.D)/(2*self.D**2)
        u = torch.rand(mu.size(),device='cuda')
        s = torch.zeros(mu.size(),device='cuda')
        s[(u<p1).nonzero(as_tuple=True)] = self.D 
        u[(u<p1).nonzero(as_tuple=True)] = 10.
        s[(u<(p1+p2)).nonzero(as_tuple=True)] = -self.D
        return s

    def fp_sample(self,var):
        p1 = var/(2*self.D**2)
        u = torch.rand(var.size(),device='cuda')
        s = torch.zeros(var.size(),device='cuda')
        ind1 = (u<p1).nonzero(as_tuple=True)
        s[ind1] = self.D[ind1]
        u[ind1] = 10.
        ind2 = (u<(2*p1)).nonzero(as_tuple=True)
        s[ind2] = -self.D[ind2]
        return s

    def fp_sample_mu(self,mu):
        p1 = (self.var_fix+mu**2+mu*self.D)/(2*self.D**2)
        p2 = (self.var_fix+mu**2-mu*self.D)/(2*self.D**2)
        u = torch.rand(mu.size(),device='cuda')
        s = torch.zeros(mu.size(),device='cuda')
        ind1 = (u<p1).nonzero(as_tuple=True)
        s[ind1] = self.D[ind1]
        u[ind1] = 10.
        ind2 = (u<(p1+p2)).nonzero(as_tuple=True)
        s[ind2] = -self.D[ind2]
        return s

    def __repr__(self):
        return "LP Optimizer: {}".format(self.optim.__repr__())

    def __str__(self):
        return "LP Optimizer: {}".format(self.optim.__str__())