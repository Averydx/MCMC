#include "MCMC.h"
#include <iostream>
void MCMC::load_data(std::string filePath)
{
	
	io::CSVReader<2> in("noisy_data.csv");
	in.read_header(io::ignore_missing_column,"time", "cell_count");
	double cell_count; double time;
	if (!in.has_column("time"))
		std::cout << "data doesn't have time" <<std::endl; 
	if (!in.has_column("cell_count"))
		std::cout << "data doesn't have cell_count" << std::endl;
 
	while (in.read_row(time,cell_count)) {
		time_series.push_back(time); 
		cell_count_data.push_back(cell_count); 
	}


}

void MCMC::init()
{
	double param_val[2];
	param_val[0] = 0;
	param_val[1] = 1;
	integrator->ode_system.params = param_val; 
	double LL_init = LL(this->integrator,cell_count_data);
	this->LL_set.push_back(LL_init); 
	this->param_set.push_back(new params(param_val)); 
	
}

void MCMC::run(int iterations)
{
	this->iterations = iterations;
	double param_val[2];
	param_val[0] = 0;
	param_val[1] = 1;
	integrator->ode_system.params = param_val;
	double LL_init = LL(this->integrator, cell_count_data);
	this->LL_set.push_back(LL_init);
	this->param_set.push_back(new params(param_val));
	double temp_scale[2]; 
	temp_scale[0] = 0.1; 
	temp_scale[1] = 5; 
	this->scale = temp_scale; 

	this->metropolis(); 

}

void MCMC::metropolis()
{
	int num_accept = 0; 
	for (int i = 1; i < this->iterations; i++)
	{
		if ((double)num_accept / (double)i > .234)
		{
			for (int j = 0; j < 2; j++)
			{
				scale[j] *= 1.1; 
			}
		}
		else
		{
			for (int j = 0; j < 2; j++)
			{
				scale[j] *= 0.9;
			}
		}

		params* last_param = param_set.at(param_set.size()-1); 
		printf("last_param: (%f,%f)\n", last_param->param0, last_param->param1); 
		

		//c++ random number generators
		std::random_device rd{};
		std::mt19937 gen{ rd() };


		std::normal_distribution<> d1{param_set.at(param_set.size() - 1)->param0, this->scale[0]};
		std::normal_distribution<> d2{ param_set.at(param_set.size() - 1)->param1, this->scale[1]};

		double paramtest[2];

		paramtest[0] = d1(gen); 
		paramtest[1] = d2(gen); 


		this->integrator->ode_system.params = (void*)paramtest; 
		double LL_test = this->LL(integrator,this->cell_count_data);
		//printf("%f\n", LL_set.back()); 

		double accept = std::min((double)1, exp(LL_test - this->LL_set.back())); 

		std::uniform_real_distribution<double> accept_distribution(0.0, 1.0);
		double prob = accept_distribution(gen);
		if (prob < accept)
		{
			num_accept+=1; 
			LL_set.push_back(LL_test); 
			param_set.push_back(new params(paramtest)); 
		}
		else
		{
			LL_set.push_back(LL_set.back());
			param_set.push_back(param_set.back());
		}


	}
}

MCMC::MCMC(Integrator* integrator, double (*LL)(Integrator* integrator, std::vector<double> data))
{
	this->integrator = integrator; 
	this->LL = LL; 
}
