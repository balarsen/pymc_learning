# http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3097064/
from pylab import *
import spacepy.plot as spp  # for the styles
import numpy as np
import pymc as pm

import pymc
import numpy as np

n = 5 * np.ones(4, dtype=int)
x = np.array([-.86, -.3, -.05, .73])

alpha = pymc.Normal('alpha', mu=0, tau=.01)
beta = pymc.Normal('beta', mu=0, tau=.01)


@pymc.deterministic
def theta(a=alpha, b=beta):
    """theta = logit^{−1}(a+b)"""
    return pymc.invlogit(a + b * x)


d = pymc.Binomial('d', n=n, p=theta, value=np.array([0., 1., 3., 5.]),
                  observed=True)

import pymc

# S = pymc.MCMC(mymodel, db = 'pickle')
S = pymc.MCMC([alpha, beta, theta], db='txt')
S.sample(iter=10000, burn=5000, thin=2)
pymc.Matplot.plot(S)
pymc.Matplot.summary_plot(S)
