
# coding: utf-8

# In[12]:

import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import tqdm
import spacepy.toolbox as tb

get_ipython().magic('matplotlib inline')


# In[2]:

# http://permalink.lanl.gov/object/tr?what=info:lanl-repo/lareport/LA-UR-93-1179


# * Shots are fired isotropically from a point and hit a position sensitive detector
# * There is no scattering
# * y is fixed to be 1 away

# In[3]:

# generate some data
with pm.Model() as model:
    x = pm.Cauchy(name='x', alpha=0, beta=1)
    trace = pm.sample(20000)
    pm.traceplot(trace)
sampledat = trace['x']


# In[4]:

trace.varnames, trace['x']
plt.hist(sampledat, 200, normed=True);
plt.yscale('log');


# In[5]:

np.random.randint(0, len(sampledat), 10)


# In[39]:

# generate some data
bins = np.linspace(-4,4,100)
hists = {}
stats = {}
for npts in tqdm.tqdm_notebook(range(1,102,40)):
    d1 = sampledat[np.random.randint(0, len(sampledat), npts)]
    with pm.Model() as model:
        alpha = pm.Uniform('loc', -10, 10)
        #     beta = pm.Uniform('dist', 1, 1)
        x = pm.Cauchy(name='x', alpha=alpha, beta=1, observed=d1)
        trace = pm.sample(10000)
        hists[npts] = np.histogram(trace['loc'], bins)
        stats[npts] = np.percentile(trace['loc'], (1, 5, 25, 50, 75, 95, 99))


# In[40]:

keys = sorted(list(hists.keys()))
for k in keys:
    p = plt.plot(tb.bin_edges_to_center(bins), hists[k][0]/np.max(hists[k][0]), 
                 drawstyle='steps', label=str(k), lw=1)
    c = p[0].get_color()
    plt.axvline(stats[k][3], lw=3, color=c)
    print(k, stats[k][2:5], stats[k][3]/(stats[k][4]-stats[k][2]), )
plt.legend()
plt.xlim((-2,2))


# ## if both are unknown

# In[65]:

# generate some data
bins = np.linspace(-4,4,100)
hists2 = {}
stats2 = {}
hists2d = {}
binsd = np.linspace(0.1,5,100)
for npts in tqdm.tqdm_notebook((1,2,5,10,20,40,60,80,200)):
    d1 = sampledat[np.random.randint(0, len(sampledat), npts)]
    with pm.Model() as model:
        alpha = pm.Uniform('loc', -10, 10)
        beta = pm.Uniform('dist', 0.1, 5)
        x = pm.Cauchy(name='x', alpha=alpha, beta=beta, observed=d1)
        trace = pm.sample(10000)
        hists2[npts] = np.histogram(trace['loc'], bins)
        stats2[npts] = np.percentile(trace['loc'], (1, 5, 25, 50, 75, 95, 99))
        hists2d[npts] = np.histogram2d(trace['loc'], trace['dist'], bins=(bins, binsd))


# In[66]:

keys = sorted(list(hists2.keys()))
for k in keys:
    p = plt.plot(tb.bin_edges_to_center(bins), hists2[k][0]/np.max(hists2[k][0]), 
                 drawstyle='steps', label=str(k), lw=1)
    c = p[0].get_color()
    plt.axvline(stats2[k][3], lw=3, color=c)
    print(k, stats2[k][2:5], stats2[k][3]/(stats2[k][4]-stats2[k][2]), )
plt.legend()
plt.xlim((-2,2))


# In[69]:

# plt.contour(hists2d[1][0], 5)
from matplotlib.colors import LogNorm

keys = sorted(list(hists2.keys()))
for k in keys:
    plt.figure()
    plt.pcolormesh(tb.bin_edges_to_center(binsd), 
                   tb.bin_edges_to_center(bins),
                   hists2d[k][0], 
                   norm=LogNorm())
    plt.title(str(k))
    plt.colorbar()
    plt.axvline(1, lw=0.5, c='k')
    plt.axhline(0, lw=0.5, c='k')


# In[ ]:



