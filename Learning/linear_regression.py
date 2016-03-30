# http://people.duke.edu/~ccc14/sta-663/PyMC2.html?highlight=invlogit

import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import spacepy.plot as spp
import pymc



# observed data
n = 21
a = 6
b = 2
sigma = 2
x = np.linspace(0, 1, n)
y_obs = a*x + b + np.random.normal(0, sigma, n)
data = pd.DataFrame(np.array([x, y_obs]).T, columns=['x', 'y'])

data.plot(x='x', y='y', kind='scatter', s=50)

# define priors
a = pymc.Normal('slope', mu=0, tau=1.0/10**2)
b = pymc.Normal('intercept', mu=0, tau=1.0/10**2)
tau = pymc.Gamma("tau", alpha=0.1, beta=0.1)

# define likelihood
@pymc.deterministic
def mu(a=a, b=b, x=x):
    return a*x + b



y = pymc.Normal('y', mu=mu, tau=tau, value=y_obs, observed=True)

# inference
m = pymc.Model([a, b, tau, x, y])
mc = pymc.MCMC(m)
mc.sample(iter=110000, burn=10000, thin=3)

abar = a.stats()['mean']
bbar = b.stats()['mean']
data.plot(x='x', y='y', kind='scatter', s=50);
xp = np.array([x.min(), x.max()])
plt.plot(a.trace()*xp[:, None] + b.trace(), c='red', alpha=0.01)
plt.plot(xp, abar*xp + bbar, linewidth=2, c='red');


pymc.Matplot.plot(mc)
pymc.Matplot.summary_plot(mc)




