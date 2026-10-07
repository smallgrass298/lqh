/*=================================================================================
 * Class Function:	Class for basic functions
 *=================================================================================*/

#ifndef FUNCTION_H
#define FUNCTION_H

/*---------------------------------------------------------------------------------
 * Standard C++ library headers
 *---------------------------------------------------------------------------------*/

#include <iostream>
#include <fstream>
#include <math.h>

/*---------------------------------------------------------------------------------
 * Local headers
 *---------------------------------------------------------------------------------*/

#include "Array.hpp"
using namespace ARRAY;
class Function
{
public:
	/* Constructor */
	Function() = default;

	/* Summation of the two 1D arrays after math operation */
	double sum(int flag, Array<double, 1> &a, Array<double, 1> &b);

	/* Destructor */
	~Function() { ; };
};
#endif
