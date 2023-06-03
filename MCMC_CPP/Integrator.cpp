
#include "Integrator.h"

Integrator::Integrator(int dim, double eps_abs, double eps_rel, double tmin, double tmax, double delta_t, int(*dfunc)(double t, const double y[], double f[], void* params_ptr))
{
	//initialize class fields 
	this->dimension = dim;
	this->eps_abs = eps_abs;
	this->eps_rel = eps_rel;
	this->tmin = tmin;
	this->tmax = tmax;
	this->delta_t = delta_t;
	this->dfunc = dfunc;

	this->drv = nullptr;

}

int Integrator::init_drv()
{
	this->drv = gsl_odeiv2_driver_alloc_y_new(&ode_system, gsl_odeiv2_step_rkf45, this->h, this->eps_abs, this->eps_rel);
	if (drv)
		return 0;
	else if (!drv)
		return EXIT_FAILURE;
}

void Integrator::init_ode_system()
{
	//initialize the fields of the ode_system struct 

	this->ode_system.function = this->dfunc;
	this->ode_system.dimension = this->dimension;
}

void Integrator::load_params(double* params)
{
	this->ode_system.params;
}


double* Integrator::Integrate()
{
	double sol[100]; 
	double y[1];	/* current solution vector */
	y[0] = 1;			/* initial value of x */
	t = tmin;
	for (t_next = tmin + delta_t; t_next <= tmax; t_next += delta_t)
	{
		status = gsl_odeiv2_driver_apply(this->drv, &t, t_next, y);
		if (status != GSL_SUCCESS) {
			printf("Error: status = %d \n", status);
			break;
		}

		double val = y[0]; 
		sol[(int)t_next - 1] = val; 
	} // end for

	return sol; 
}

