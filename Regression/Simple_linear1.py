
# coding: utf-8

# In[14]:

# https://dansaber.wordpress.com/2014/05/28/bayesian-regression-with-pymc-a-brief-tutorial/


# In[17]:

import pymc 
import random

import numpy as np
import matplotlib.pyplot as plt
import spacepy.plot as spp

from scipy.stats import multivariate_normal


# In[45]:

float_df = {}
float_df['weight'] = np.linspace(10,100, 20)
float_df['mpg'] = np.linspace(30,60, 20) * (np.random.random_sample(size=20)*10) + float_df['weight']*10
plt.scatter(float_df['weight'], float_df['mpg'])
plt.xlabel('weight')
plt.ylabel('mpg')


# In[46]:

# NOTE: the linear regression model we're trying to solve for is
# given by:
# y = b0 + b1(x) + error
# where b0 is the intercept term, b1 is the slope, and error is
# the error
 
# model the intercept/slope terms of our model as
# normal random variables with comically large variances
b0 = pymc.Normal('b0', 0, 0.0003)
b1 = pymc.Normal('b1', 0, 0.0003)
 
# model our error term as a uniform random variable
err = pymc.Uniform('err', 0, 500)
 
# "model" the observed x values as a normal random variable
# in reality, because x is observed, it doesn't actually matter
# how we choose to model x -- PyMC isn't going to change x's values
x_weight = pymc.Normal('weight', 0, 1, value=np.array(float_df['weight']), observed=True)
 
# this is the heart of our model: given our b0, b1 and our x observations, we want
# to predict y
@pymc.deterministic
def pred(b0=b0, b1=b1, x=x_weight):
    return b0 + b1*x
 
# "model" the observed y values: again, I reiterate that PyMC treats y as
# evidence -- as fixed; it's going to use this as evidence in updating our belief
# about the "unobserved" parameters (b0, b1, and err), which are the
# things we're interested in inferring after all
y = pymc.Normal('y', pred, err, value=np.array(float_df['mpg']), observed=True)
 
# put everything we've modeled into a PyMC model
model = pymc.Model([pred, b0, b1, y, err, x_weight])


# In[ ]:

mcmc = pymc.MCMC(model)
mcmc.sample(50000, 20000, thin=60)


# In[ ]:

pymc.Matplot.plot(mcmc)


# In[ ]:

b0.trace().shape



# In[ ]:

# the data
plt.scatter(float_df['weight'], float_df['mpg'])
plt.xlabel('weight')
plt.ylabel('mpg')

xtmp = float_df['weight']
for i in range(20):
    ind = np.random.randint(0, b0.trace().shape[0], 1)
    ytmp = b0.trace()[ind] + xtmp*b1.trace()[ind] + err.trace()[ind]
    plt.plot(xtmp, ytmp, lw=0.5)


# In[ ]:



