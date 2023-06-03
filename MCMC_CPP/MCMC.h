#pragma once
#include <string>
#include "csv.h"
#include "Integrator.h"
#include <random>

struct params
{
	double param0; 
	double param1; 

	params(double* param_vals)
	{
		param0 = param_vals[0]; 
		param1 = param_vals[1]; 
	}
};


class MCMC
{
public: 

	//fields
	std::vector<double> time_series; 
	std::vector<double> cell_count_data; 
	Integrator* integrator; 
	double (*LL)(Integrator* integrator,std::vector<double> data);
	std::vector<params*> param_set; 
	std::vector<double> LL_set; 
	int iterations; 
	double* scale; 
	

	//functions 
	MCMC(Integrator* integrator, double (*LL)(Integrator* integrator, std::vector<double> data));
	void load_data(std::string filePath);
	void init(); 
	void run(int iterations); 
	void metropolis(); 
};

