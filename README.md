# MCMC

Object oriented approach to the MCMC testing library. 

Steps to set up a test: 

1. Create a "MCMC" instance. This class controls the running of the sim itself and the loading of initial conditions, ODE model, and Log Likelihood. In the constructor pass in the ODE model and the Log Likelihood function to optimize. These must be function pointers with a specific parameter set. See example.    

2. Pass a initial condition using the MCMC:load_initial_condition function . 


3. Create a "data_gen" instance. This class controls the input data to the algorithm. Use the function "load_data" and pass a .csv file and a sampler to pass the data to. 

4. Call sampler, pass the number of iterations to run as an argument.


Output will be of this form. Top is a fit of a curve to the data set where green is the most current parameter set. Bottom is a trace plot of the Log likelihood proposals over time. 

![distribution_data](https://github.com/Averydx/MCMC/assets/79957750/8a206383-bff0-42fd-8084-71991edafecb)




