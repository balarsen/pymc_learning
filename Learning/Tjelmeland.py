# Introduction to Markov chain Monte Carlo — with examples ...
# https://www.researchgate.net/file.PostFileLoader.html?id=515196a8d039b13015000002&assetKey=AS%3A271835466272768%401441822033782

from pylab import *

import pymc
from pymc import Matplot
import numpy as np
from scipy.misc import factorial
import spacepy.plot as spp

data=np.array([33,66,1])
rates=pymc.Uniform('rates',0,100,size=4,value=[0.01,2,10,1])

@pymc.deterministic(plot=True)
def prob(rates=rates):
    return np.array([0.33,0.66,0.01])

likelihood=pymc.Multinomial('likelihood',n=sum(data),p=prob,value=data,observed=True)
M = pymc.MCMC(likelihood)

M.sample(100000)

Matplot.summary_plot(M)

#
# @pymc.observed
# def y(value=1):
#     pymc.categorical_like()
#
#     return 10**value * np.exp(-10)/ factorial(value)
#
# M = pymc.MCMC(y)




