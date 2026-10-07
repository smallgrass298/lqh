/*=================================================================================
 * Class TimeAdv:	The time advance scheme:
 *					1st-order Euler Explicit Scheme
 *					3st-order TVD Runge-Kutta Scheme
 *=================================================================================*/

#ifndef TIME
#define TIME

#include "Array.hpp"
using namespace ARRAY;

class TimeAdv
{
public:
	/* Constructor */
	TimeAdv() = default;

	/* 1st-order Euler Explicit Scheme */
	void EE(int, double, Array<double, 1>&, Array<double, 1>&,Array<double, 1>&, int, Array<double, 4>&, Array<double, 4>&,Array<double, 4>&, Array<double, 4> &);

	/* 3st-order TVD Runge-Kutta Scheme */
	void TVD_RK3(int, double, Array<double, 1>&, Array<double, 1>&,Array<double, 1>&, int, Array<double, 4>&, Array<double, 4>&,Array<double, 4>&, Array<double, 4> &);

	/* Destructor */
	~TimeAdv() { ; };
};

#endif
