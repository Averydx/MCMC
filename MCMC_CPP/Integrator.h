#pragma once
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <vector>


class Integrator
{
public:
	int dimension;   /* number of differential equations */
	int status;                /* status of driver function */
	double eps_abs;  /* absolute error requested  */
	double eps_rel;

	double* y;	        /* current solution vector */

	double t, t_next;             /* current and next independent variable */
	double tmin, tmax, delta_t; /* range of t and step size for output */

	double h = 1.0e-6;            /* starting step size for ode solver */

	//function to integrate
	int (*dfunc)(double t, const double y[], double f[], void* params_ptr);

	gsl_odeiv2_driver* drv;

	gsl_odeiv2_system ode_system;

	//Constructor to initialize all fields 
	Integrator(int dim, double eps_abs, double eps_rel, double tmin, double tmax, double delta_t, int (*dfunc)(double t, const double y[], double f[], void* params_ptr));
	//runs the integrator
	double* Integrate();

	int init_drv();

	void init_ode_system();

	void load_params(double* params); 

};

