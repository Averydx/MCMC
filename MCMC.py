from random import random

from pandas.core.tools.datetimes import Tuple
from scipy.integrate import odeint;
from scipy.stats import norm;
import numpy as np;
import matplotlib.pyplot as plt
import pandas as pd;
import time;
from random import randrange;

class MCMC:
    #Takes 2 arguments, an ODE and a log likelihood function to optimize
    def __init__(self,ODE,LL):
        self.data = None
        self.time_series = None
        self.ODE = ODE;
        self.LL = LL;

    def print_data(self):
        print(self.data);

    def load_intial_condition(self,ODE_0):
        self.intial_condition = ODE_0;

    def init_fields(self):
        # arbitrary first guess for parameter set
        params_0 = [0, 1];

        # first LL
        LL_init = self.LL(self.ODE,self.intial_condition,self.time_series,self.data, params_0);

        # tracks all tested parameter sets--appends the initial guess to start
        self.param_set = [];
        self.param_set.append(params_0);

        # tracks the log likelihood of the param set
        self.LL_set = [];
        self.LL_set.append(LL_init);

        # tracks all param guesses--even if rejected
        self.all_guesses = [];
        self.all_guesses.append(params_0);

    def metropolis(self):
        num_accept = 0;
        for i in range(1, self.iter):
            # monitor progress
            if i % 100 == 0:
                print("Percent complete: {0}".format(round((i / self.iter) * 100, 2)));
                print("acceptance rate: {0}\n".format(round((num_accept / i) * 100, 2)));
                if num_accept/i > .234:
                    self.scale *= 1.1;
                else:
                    self.scale *= 0.9;

            paramtest = [np.random.normal(loc=self.param_set[-1][0], scale=self.scale[0]),
                         np.random.normal(loc=self.param_set[-1][1], scale=self.scale[1])];

            #compute the log likelihood of the paramtest given the data
            LL_test = self.LL(self.ODE,self.intial_condition,self.time_series,self.data, paramtest);

            #acceptance criteria
            accept = min(1, np.exp(LL_test - self.LL_set[-1]))
            #acceptance check
            if np.random.random() < accept:
                num_accept += 1;
                self.LL_set.append(LL_test);
                self.param_set.append(paramtest);
            else:
                self.LL_set.append(self.LL_set[-1]);
                self.param_set.append(self.param_set[-1]);

            self.all_guesses.append(self.param_set[-1]);

    def plot(self):


        fig, (ax1, ax2) = plt.subplots(2);
        ax1.scatter(self.data['time'], self.data['cell_count'], marker='o', s=5);
        # plots some previous param sets for visualization
        for i in range(0, 100):
            rand = randrange(0, len(self.param_set));
            ax1.plot(self.time_series,
                     odeint(self.ODE, self.intial_condition, self.time_series, args=tuple(self.param_set[rand])),
                     color="r", alpha=0.1);
        ax1.plot(self.time_series,
                 odeint(self.ODE, self.intial_condition, self.time_series, args=tuple(self.param_set[-1])),
                 color="green", alpha=1);

        ax2.plot(self.LL_set);


        plt.show();

    #runs the sampler--arg is number of iterations to run
    def run(self,iter):
       self.iter = iter;
       self.init_fields();
       #scale values for the variance of the prior distributions
       self.scale = np.array([0.1,7]);

       #Metropolis algorithm
       self.metropolis();
       self.plot();

class data_gen:
    def generate_data(self,ODE,ODE_0,time):
        params = np.array([np.random.uniform(0,1), np.random.uniform(500,2000)]);
        print(params)
        X = odeint(ODE, ODE_0, time, args=tuple(params));
        for x in range(len(X)):
            X[x] += np.random.normal(0, 100);

        X = np.squeeze(X);
        data = np.stack((time, X), axis=-1);

        df = pd.DataFrame(data);
        df.columns = ["time", "cell_count"];
        df.to_csv('noisy_data.csv', index=False);

    #Takes in a file_path and returns a pandas dataframe containing your data
    def load_data(self,sampler,file_path):
        sampler.data = pd.read_csv(file_path,sep=',');

        # TODO test to make sure column is named time
        sampler.time_series = sampler.data['time'];



#model definitions

#ODE model
def model(X,t,arg1,arg2):
  r = arg1;
  k = arg2;
  dXdt = r*X*(1-X/k);
  return dXdt;

#Log likelihood
def log_likelihood(ODE,ODE_0, t, data,params):
  test_data = odeint(ODE,ODE_0,t,args=tuple(params));
  LL = 0;
  for i in range(len(data['time'])-1):
    nd = norm(loc=test_data[i],scale = 1000);
    LL += nd.logpdf(data['cell_count'][i]);
  return LL;


time_series = np.arange(0,100,1);
generator = data_gen();
generator.generate_data(model,1,time_series);
sampler = MCMC(model,log_likelihood);
sampler.load_intial_condition(1);
generator.load_data(sampler,"noisy_data.csv");
sampler.print_data();

sampler.run(1000);








