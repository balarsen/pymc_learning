# http://people.duke.edu/~ccc14/sta-663/PyMC2.html?highlight=invlogit

import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import spacepy.plot as spp
import pymc

n = 100
h = 61
alpha = 2
beta = 2

p = pymc.Beta('p', alpha=alpha, beta=beta)
y = pymc.Binomial('y', n=n, p=p, value=h, observed=True)
m = pymc.Model([p, y])

mc = pymc.MCMC(m, )
mc.sample(iter=110000, burn=10000)

# plt.hist(p.trace(), 15, histtype='step', normed=True, label='post');
# x = np.linspace(0, 1, 100)
# plt.plot(x, stats.beta.pdf(x, alpha, beta), label='prior');
# plt.legend(loc='best');

pymc.Matplot.plot(mc)
pymc.Matplot.summary_plot(mc)

