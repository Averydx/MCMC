
#define _CRTDBG_MAP_ALLOC

#include <iostream>
#include <gsl/gsl_errno.h>
#include "Integrator.h"
#include "MCMC.h"
#include <time.h>
#include <gsl/gsl_randist.h>
#include <random>
#include <math.h>
#include "stats.hpp"
#include <crtdbg.h>

int dfunc(double t, const double y[], double f[], void* params_ptr); 
double LL(Integrator* integrator,std::vector<double>data); 

int main()
{
	clock_t begin = clock();


	Integrator* integrator = new Integrator(1, 1.e-8, 1.e-10, 0, 100, 1, dfunc); 

	integrator->init_ode_system(); 
	
	integrator->init_drv(); 
	MCMC sampler(integrator,LL); 
	sampler.load_data("noisy_data.csv");
	double params[2] = {0,1};
	sampler.integrator->ode_system.params = params; 
	sampler.iterations = 100; 
	double scale[2] = { 0.1,7 }; 
	sampler.scale = scale; 
	sampler.param_set.push_back(new struct params(params)); 
	sampler.LL_set.push_back(LL(sampler.integrator, sampler.cell_count_data));
	int num_accept = 0;
	for (int i = 1; i < sampler.iterations; i++)
	{
		
		double accept_rate = (double)num_accept / (double)i; 
		printf("acceptance rate:%f\n", accept_rate);
		//if (accept_rate > .234)
		//	for (int j = 0; j < 2; j++)
		//		sampler.scale[j] *= 1.1; 
		//else
		//	for (int j = 0; j < 2; j++)
		//		sampler.scale[j] *= 0.9;

		std::random_device rd{};
		std::mt19937 gen{ rd() };


		std::normal_distribution<> d1{ sampler.param_set.at(sampler.param_set.size() - 1)->param0, sampler.scale[0] };
		std::normal_distribution<> d2{ sampler.param_set.at(sampler.param_set.size() - 1)->param1, sampler.scale[1] };

		double paramtest[2];

		paramtest[0] = d1(gen);
		paramtest[1] = d2(gen);

		while (paramtest[0] < 0 || paramtest[0] > 1)
			paramtest[0] = d1(gen); 

		while (paramtest[1] < 0||paramtest[1] > 10000)
			paramtest[1] = d2(gen);


		printf("(%f,%f)\n", paramtest[0], paramtest[1]); 
		sampler.integrator->ode_system.params = paramtest;

		double LL_test = sampler.LL(sampler.integrator, sampler.cell_count_data);

		double accept = std::min((double)1, exp(LL_test - sampler.LL_set.at(sampler.LL_set.size() - 1)));

		std::uniform_real_distribution<double> accept_distribution(0.0, 1.0);
		double prob = accept_distribution(gen);

		printf("Log Likelihood: %f\n", LL_test); 

		if (prob < accept)
		{
			num_accept += 1;
			sampler.LL_set.push_back(LL_test);
			sampler.param_set.push_back(new struct params(paramtest));
		}
		else
		{
			sampler.LL_set.push_back(sampler.LL_set.at(sampler.LL_set.size() - 1));
			sampler.param_set.push_back(sampler.param_set.at(sampler.param_set.size() - 1));
		}

		printf("\n"); 
	}

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("time spent: %f\n", time_spent); 
	_CrtDumpMemoryLeaks(); 
	return 0;
}

int dfunc(double t, const double y[], double f[], void* params_ptr) {
	double* lparams = (double*)params_ptr;
	/* get parameter(s) from params_ptr */
	double alpha = lparams[0];
	double omega = lparams[1];
	/* evaluate the right-hand-side functions at t */
	f[0] = alpha * y[0] * (1 - y[0]/omega);
	return GSL_SUCCESS; /* GSL_SUCCESS defined in gsl/errno.h as 0 */
}

double LL(Integrator* integrator,std::vector<double> data)
{
	double* test_data = integrator->Integrate();
	double LogL_sum = 0; 
	for (int i = 0; i < integrator->tmax-1; i++)
	{
		LogL_sum += stats::dnorm(data.at(i), test_data[i], 1000, true);
	}
	return LogL_sum;
}



