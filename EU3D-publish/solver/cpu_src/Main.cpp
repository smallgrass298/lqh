/*=================================================================================
 * This program is to solve the 2D multi-species Euler equation.
 *
 * For User:
 * Open the ctrl.txt:	Set the ctrl parameters
 * Open the input.txt:	Set the initial parameters and chemical reaction model
 * Open the grid.txt:	Set the grid parameters
 *=================================================================================*/

/*---------------------------------------------------------------------------------
 * Standard C++ library headers
 *---------------------------------------------------------------------------------*/

#include <iostream>
#include <mpi.h>
// #include <omp.h>
// #include <vector>
// #include <tuple>
// #include <random>

/*---------------------------------------------------------------------------------
 * Local headers
 *---------------------------------------------------------------------------------*/

#include "Euler.hpp"
#include "StopWatch.hpp"
#include "TimeBase.hpp"
// #include <Array.hpp>
// #include "LoadBalance.hpp"
/*---------------------------------------------------------------------------------
 * Main function
 *---------------------------------------------------------------------------------*/
using namespace std;
int main()
{
	/* Record the running time */
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	cout << "Total process: " << numprocs << endl;

	int t1 = MPI_Wtime();
	/* Solve the equation */
	Euler value;
	int t2 = MPI_Wtime();
	if (myid == 0)
	{
		cout << "Total Time: " << t2 - t1 << endl;
	}

	MPI_Finalize();

	return 0;
}