
# coding: utf-8

# In[3]:

import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import tqdm
import spacepy.toolbox as tb

get_ipython().magic('matplotlib inline')


# # Setup a convolution data set and try and doconvolve it
# 

# In[39]:

np.random.seed(8675309)

dat_len = 100
xval = np.arange(dat_len)
realdat = np.zeros(dat_len, dtype=int)
realdat[40:60] = 50
noisemean = 2
real_n = np.zeros_like(realdat)
for i in range(len(realdat)):
    real_n[i] = np.random.poisson(realdat[i]+noisemean)


# make a detector
# triangular with FWFM 5 and is square
det = np.array([1,1,1,1,1])

# the numpy convolve I don't understand the normalization
obs = np.convolve(real_n, det, mode='same')
obs = tb.normalize(obs)
obs *= real_n.max()
    
# Two subplots, the axes array is 1-d
f, axarr = plt.subplots(4, sharex=True)
axarr[0].plot(xval, realdat)
axarr[0].set_ylabel('Truth')
axarr[0].set_ylim((0,60))

axarr[1].plot(xval, real_n)
axarr[1].set_ylabel('T+N')

axarr[2].plot(np.arange(len(det)), det)
axarr[2].set_ylabel('Det')

axarr[3].plot(xval, obs)
axarr[3].set_ylabel('Obs')




# So the det provides a point spread function that is U(0,5)

# In[44]:

# generate some data
with pm.Model() as model:
    truth_mc = pm.Uniform('truth', 0, 100, shape=dat_len)
    noisemean_mc = pm.Uniform('noisemean', 0, 100)
    noise_mc = pm.Poisson('noise', noisemean_mc, observed=obs[1:20])
    real_n_mc = pm.Poisson('real_n', truth_mc+noisemean_mc, shape=dat_len)
    psf = pm.Uniform('psf', 0, 5, observed=det)
    obs_mc = pm.Normal('obs', (truth_mc+noisemean_mc)*psf.max(), 1/5**2, observed=obs, shape=dat_len)
    
    trace = pm.sample(5000)



# In[45]:

pm.traceplot(trace)


# In[46]:

pm.summary(trace)


# In[50]:

# plt.plot(trace['truth'][0:5,:].T)
trace['truth'].shape
iqr = np.zeros((dat_len,3))
for i in range(dat_len):
    iqr[i] = np.percentile(trace['truth'].T[i], (25,50,75), axis=0)
plt.plot(xval, iqr[:,1],  label='recovered')
plt.fill_between(xval, iqr[:,0], iqr[:,2], alpha=0.2)
plt.plot(xval, real_n, c='r', label='pre psf')
plt.plot(xval, realdat, c='g',  label='truth')
plt.plot(xval, obs, c='k',  label='observed', lw=3)

plt.legend()
plt.figure()
snr = iqr[:,1]/(iqr[:,2], iqr[:,0])
perixval.shape, snr.shape
plt.plot(xval, snr)


# In[48]:

print(np.percentile(trace['noisemean'], (25,50,75)), noisemean)


# In[49]:

obs[1:20]


# In[ ]:



