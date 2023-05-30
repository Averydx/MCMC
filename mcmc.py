from random import random

from pandas.core.tools.datetimes import Tuple
from scipy.integrate import odeint;
from scipy.stats import norm;
import numpy as np;
import matplotlib.pyplot as plt
import pandas as pd;
import time;
from random import randrange;

#ODE model
def model(X,t,arg1,arg2):
  r = arg1;
  k = arg2;
  dXdt = r*X*(1-X/k);
  return dXdt;

x0 = 1;
t = np.arange(0,100,1);

#Stochastic data
experimental_data= pd.read_csv("cell_count_data.csv",sep=',');

#Log likelihood
def log_likelihood(data,params):
  x0 = 1;
  t = np.arange(0,100,1);
  test_data = odeint(model,x0,t,args=tuple(params));
  LL = 0;
  for i in range(len(data['time'])-1):
    nd = norm(loc=test_data[i],scale = 1000);
    LL += nd.logpdf(data['cell_count'][i]);
  return LL;

######################


#MCMC

#first params guess
params_0 = [0,1];

#first LL
LL_init = log_likelihood(experimental_data,params_0);
param_set = [];
param_set.append(params_0);

LL_set = [];
LL_set.append(LL_init);

#tracks all param guesses--even if rejected
all_guesses = [];
all_guesses.append(params_0);

iter_count = 10000;
time_0 = time.process_time();
num_accept = 0;
scale = np.array([0.1,7]);

for i in range(1,iter_count):
  #monitor progress
  if i % 100 == 0:
    print("Percent complete: {0}".format(round((i/iter_count)*100,2)));
    print("acceptance rate: {0}\n".format(round((num_accept/i)*100,2)));

  paramtest = [np.random.normal(loc = param_set[-1][0], scale = scale[0]),np.random.normal(loc = param_set[-1][1], scale = scale[1])];

  LL_test = log_likelihood(experimental_data,paramtest);



  #perform MCMC--metropolis hastings



  accept = min(1,np.exp(LL_test - LL_set[-1]))

  if np.random.random() < accept:
    num_accept +=1;
    LL_set.append(LL_test);
    param_set.append(paramtest);
  else:
    LL_set.append(LL_set[-1]);
    param_set.append(param_set[-1]);

  all_guesses.append(param_set[-1]);

time_1 = time.process_time();

print("Time in seconds: {0}".format(time_1 - time_0));

#####################

#Plotting



plt.scatter(experimental_data['time'],experimental_data['cell_count'], marker='o', s = 5);

for i in range(0,100):

  rand = randrange(0,len(param_set));
  plt.plot(t,odeint(model,x0,t,args=tuple(param_set[rand])), color="r", alpha=0.1);


print("percent of parameters sets accepted: {0}".format((num_accept/iter_count)*100));

print(param_set[-1]);

plt.plot(t,odeint(model,x0,t,args=tuple(param_set[-1])), color="green", alpha=1);

#plt.plot(LL_set);

plt.show();

