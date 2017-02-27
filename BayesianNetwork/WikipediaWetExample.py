
# coding: utf-8

# # Example Bayesian network from wikipedia
# https://en.wikipedia.org/wiki/Bayesian_network

# In[1]:

from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "SimpleBayesNet.png")


# Suppose that there are two events which could cause grass to be wet: either the sprinkler is on or it's raining. Also, suppose that the rain has a direct effect on the use of the sprinkler (namely that when it rains, the sprinkler is usually not turned on). Then the situation can be modeled with a Bayesian network (shown to the right). All three variables have two possible values, T (for true) and F (for false).
# 
# The joimt probabilty function is:
# 
# $Pr(G,S,R)=Pr(G|S,R)Pr(S|R)Pr(R)$
# 
# where the names of the variables have been abbreviated to G = Grass wet (yes/no), S = Sprinkler turned on (yes/no), and R = Raining (yes/no).

# The model can answer questions like "What is the probability that it is raining, given the grass is wet?" by using the conditional probability formula and summing over all nuisance variables:
# 
# $Pr(R=T|G=T)=\frac{Pr(G=T|R=T}{Pr(G=T)}=\frac{\sum_{S\in  {T,F}}Pr(G=T,S,R=T)}{\sum_{S,R\in {T,F}}Pr(G=T,S,R)}$
# 
# Using the expansion for the joint probability function {\displaystyle \Pr(G,S,R)} {\displaystyle \Pr(G,S,R)} and the conditional probabilities from the conditional probability tables (CPTs) stated in the diagram, one can evaluate each term in the sums in the numerator and denominator. For example,
# 
# $Pr(G=T,S=T,R=T)=Pr(G=T|S=T,R=T)Pr(S=T|R=T)Pr(R=T)\\
# = 0.99 \times 0.01 \times 0.2\\
# =0.00198$
# 
# Then the numerical results (subscripted by the associated variable values) are
# 
# $Pr(R=T|G=T)=\frac{0.00198_{TTT} + 0.1584_{TFT}}{0.00198_{TTT}+0.288_{TTF}+0.1584_{TFF}}=\frac{891}{2491}=35.77\%$

# ## Do this with pymc3
# 

# In[2]:

import pymc3 as mc3
import numpy as np


# In[ ]:



