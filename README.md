# MCMC

Object oriented approach to the MCMC testing library. 

Steps to set up a test: 

1.Create a "MCMC" instance. This class controls the running of the sim itself and the loading of initial conditions, ODE model, and Log Likelihood. In the constructor pass in the ODE model and the Log Likelihood function to optimize.   
2. Pass a initial condition using the MCMC:load_initial_condition function  
3. Create a "data_gen" instance. This class controls the input data to the algorithm. Use the function "load_data" and pass a .csv file and a sampler to pass the data to. 


