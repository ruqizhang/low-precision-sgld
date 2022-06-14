import argparse
import time
import torch
import torch.nn.functional as F
from qtorch.quant import *
from qtorch import BlockFloatingPoint, FixedPoint, FloatingPoint
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

parser = argparse.ArgumentParser(description="SGLD training")
parser.add_argument(
    "--quant_type", type=str, default="f", help="f: full-precision gradient accumulators; \
    vc: low-precision gradient accumulators with variance-corrected quantization; \
    naive: naive low-precision gradient accumulators")
args = parser.parse_args()

def main(alpha,theta):
    var = 2*alpha
    if args.quant_type == 'vc':
        mu = theta - alpha*quant_s(grad(theta))
        theta = Q_vc(mu, var)
    elif args.quant_type == 'f':
        theta = theta - alpha*quant_s(grad(quant_s(theta))) + var**.5*torch.randn(1)
    elif args.quant_type == 'naive':
        theta = quant_s(theta - alpha*quant_s(grad(theta)) + var**.5*torch.randn(1))
    return theta

def Q_vc(mu, var):
    if var>var_fix:
        x = mu + (var-var_fix)**.5*torch.randn(1)
        quant_x = quant_n(x)
        residual = x - quant_x
        theta = quant_x + torch.sign(residual)*sample_mu(torch.abs(residual))
    else:
        quant_mu = quant_s(mu)
        residual = mu - quant_mu
        p1 = torch.abs(residual)/D
        var_s = (1.-p1)*residual**2+p1*(-residual+torch.sign(residual)*D)**2
        if var>var_s:
            theta = quant_mu + sample(var-var_s)
        else:
            theta = quant_mu #this line should not be used often, otherwise the variance will be larger than the truth
    theta = torch.clamp(theta, min=-2**(WL-FL-1), max=2**(WL-FL-1)-2**(-FL))
    return theta

def grad(theta):
    return theta + sigma*torch.randn(1)

def sample(var):
    p1 = var/(2*D**2)
    p2 = p1
    u = torch.rand(1)
    if u<p1:
        return D 
    elif u<p1+p2:
        return -D 
    else:
        return 0

def sample_mu(mu):
    p1 = (var_fix+mu**2+mu*D)/(2*D**2)
    p2 = (var_fix+mu**2-mu*D)/(2*D**2)
    u = torch.rand(1)
    if u<p1:
        return D 
    elif u<p1+p2:
        return -D 
    else:
        return 0

WL = 8
FL = 3
number = FixedPoint(wl=WL, fl=FL)
quant_s = quantizer(
    forward_number=number, forward_rounding="stochastic"
)
quant_n = quantizer(
    forward_number=number, forward_rounding="nearest"
)
theta = torch.zeros(1)
alpha = 2e-3
sigma = 0.1
D = 1./(2**FL)
var_fix = D**2/4.
theta_list = []
iteration = 10000000
for i in range(iteration):
    theta = main(alpha,theta) 
    theta_list.append(theta)
theta_list = torch.cat(theta_list)
sns.histplot(theta_list,stat='density',bins=np.arange(-5+D/2.,5,D),common_norm=False,color='r')
x_axis = np.arange(-5, 5, 0.001)
plt.plot(x_axis, norm.pdf(x_axis,0,1),label="True",lw=2,color='k')
plt.legend(fontsize=18)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.ylabel('Density',fontsize=17)
plt.savefig('figs/gaussian_%s.pdf'%(args.quant_type))